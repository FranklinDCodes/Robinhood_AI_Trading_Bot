# |---------------------------{ IMPORTS }---------------------------|
import torch as pyt
import torch.nn as nn
import torch.optim as optim
import market_data_LSTM as md
from math import sqrt
from statistics import mean, median, StatisticsError
from datetime import datetime
from random import choice
from os import path



# |---------------------------{ NEURAL NETWORK OBJECT }---------------------------|
class MARKET_LSTM(nn.Module):

    def __init__(self):
        # NN
        super().__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = pyt.nn.Sequential(
            nn.Linear(128, 1)
            )
        # Settings
        self.input_candles = 200
        self.output_minutes = 1440
        self.profit_aim = 0.01
        self.stop_loss = None
        self.stock_list = []

    # Forward pass
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x



# |---------------------------{ CONSTANTS }---------------------------|

# training epochs per network
epochs = 4

# number of training examples to use for RMSE calculation
test_size = 1000

# number of training examples per stock to exclude from the historical data for running a profitability trial later
TRIAL_SIZE = 100

# the price increase that the neural network is looking to predict
PROFIT_AIM = 0.01

# brackets classifying the network output from 0 -> 1
BUY_TRIGGERS = [round(1 - i*.1, 2) for i in range(10)]

# network learning rate per epoch (index)
LEARNING_RATES = [1e-4, 1e-5, 1e-6, 1e-7]

# the number of stocks' data to train each neural network on
STOCK_COUNT_PER_NET = 50

# stop loss
STOP_LOSS = None

# directory to save neural networks to
NETS_DIR = "nets"



# |---------------------------{ MARKET DATA SETTINGS }---------------------------|

# the number of candlesticks to input per training example
md.INPUT_CANDLE_COUNT = 200

# the time period being predicted by the network in minutes
md.OUTPUT_MINUTES = 1440

# the time period being prediction day long candlesticks (e.g. 1 for 1440 output minutes)
md.OUTPUT_CANDLES = int(md.OUTPUT_MINUTES/md.DATA_INTERVAL_MINS)

# do add gaussian noise to input
md.ADD_NOISE = True

# the number of gaussian noise values to add to each input vector
md.NOISE_COUNT = 1

# the length of the volume moving average
md.VOLUME_EMA_LEN = 14



# |---------------------------{ STOCK LISTS }---------------------------|

# takes in a list and splits it into sub-lists of a certain size or smaller
def split_list(lst, sub_list_size):

    sub_lists = []

    last_list_ind = 0

    sub_list_count = len(lst)//sub_list_size

    for part in range(1, sub_list_count + 1):

        sub_lists.append(lst[last_list_ind: sub_list_size*part])

        last_list_ind = sub_list_size*part
    
    return sub_lists

# all of the stock symbols that there is data on in the database
tables = list(md.DBO.table.keys())

# lists of stocks for the different neural networks to train on
table_lists = split_list(tables, STOCK_COUNT_PER_NET)



# |---------------------------{ TRAINING }---------------------------|

# count neural networks
net_ind = 0

# open list of stocks to train a neural network
for table_list in table_lists:
    
    # net
    model = MARKET_LSTM()
    model.stock_list = table_list.copy()

    # start training stopwatch
    model_start_time = datetime.now()

    # start data collection stopwatch
    data_start_time = datetime.now()

    # training dataset
    dataset = []

    # dataset for profitability trial
    trial_set = []


    #   <------------- Generate Datasets ------------->
    # stock ticker index
    ind = 0

    # open stock tickers
    for i in table_list:
        
        # update set market data file 
        md.DATA_TABLE = i
        md.DATA_TABLE_LEN = len(md.DBO.table[md.DATA_TABLE].getAll())

        # set a stock id for the input
        md.ID = ind/len(table_list)

        # generate training data
        count = md.getTrainingExampleCount()
        new_data = [md.getTrainingExample(i, True, PROFIT_AIM, STOP_LOSS) for i in range(count)]# - TRIAL_SIZE)]
        dataset.append(new_data)

        # generate trial data
        trial_set.append([md.getTrainingExample(i, True, PROFIT_AIM, STOP_LOSS) for i in range(count - TRIAL_SIZE, count)])

        # update index
        ind += 1
    
    # pick random test set
    test_set = [choice(choice(dataset)) for i in range(test_size)]

    print(F"DATA GATHERED after {datetime.now() - data_start_time}")
    

    #   <------------- Train ------------->
    
    # open epochs
    for epoch in range(epochs):

        # set PyTorch training tools
        optimizer = optim.Adam(model.parameters(), LEARNING_RATES[epoch])
        loss_fn = nn.MSELoss()

        # set train mode
        model.train()

        # open dataset[stocks]
        for stock in dataset:

            # open stock[examples]
            for eg in stock:

                # forward
                seq = eg[0]
                y = eg[1]
                y_pred = model(seq)

                # backward
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epoch += 1


        #   <------------- TEST ------------->
        # optimize testing efficiency
        model.eval()
        with pyt.no_grad():

            # create list of RMSE losses
            losses = []

            # open test_set training examples
            for i in test_set:

                # forward
                seq = i[0]
                y = i[1]
                y_pred = model(seq)

                # add square root of MSE to list
                test_rmse = float(sqrt(loss_fn(y_pred, y)))
                losses.append(test_rmse)
            
            # print
            print(f"epoch {epoch} | median RMSE : {median(losses)}")
    


    # |---------------------------{ PROFITABILITY TRIAL }---------------------------|

    # open bracket for different hypothetical network outputs that could trigger a purchase
    for BUY_TRIGGER in BUY_TRIGGERS:

        # trial results
        net_results = {'wins': 0, 'losses': 0, 'missed wins': 0, 'missed losses': 0}

        # the exact decreases of value for the network's losses
        loss_amounts = []
        
        # trial set stock index
        ind = 0

        # open trial-set[stocks]
        for stock in trial_set:

            # open stock[training_examples]
            for i in stock:
                
                # forward pass
                seq, labe, loss_amount = i
                out = model(seq)[-1][0]

                # label
                labe = labe[-1][0]

                # check if the network simulated a profitable purchase, or a "win"
                if out >= BUY_TRIGGER and labe >= PROFIT_AIM:
                    net_results['wins'] += 1
                
                # check if the network simulated a unprofitable purchase, or a "loss"
                elif out >= BUY_TRIGGER and labe <= PROFIT_AIM:
                    net_results['losses'] += 1
                    
                    # add exact amount loss (percentage) to loss_amount
                    loss_amounts.append(loss_amount)

                # check if the network would've missed a profitable purchase
                elif out <= BUY_TRIGGER and labe >= PROFIT_AIM:
                    net_results['missed wins'] += 1

                # check if the network would've missed a unprofitable purchase
                elif out <= BUY_TRIGGER and labe <= PROFIT_AIM:
                    net_results['missed losses'] += 1

        # calculate average loss amount
        try:
            net_results['average loss'] = round(mean(loss_amounts), 6)
        except StatisticsError:
            net_results["average loss"] = 0

        # calculate the available wins that were missed out on as a percentage of all possible purchases
        try:
            net_results["overall win percentage"] = round(net_results['missed wins']/(net_results['missed losses'] + net_results['missed wins']), 6)
        except ZeroDivisionError:
            net_results["overall win percentage"] = 0

        # calculate the percent of purchases that were wins
        try:
            net_results["prediction win percentage"] = round(net_results['wins']/(net_results['losses'] + net_results['wins']), 6)
        except ZeroDivisionError:
            net_results["prediction win percentage"] = 0

        # calculate the purchases per opportunity to purchase as a percentage
        try:
            net_results["purchase occurrence"] = round((net_results['wins'] + net_results['losses'])/sum([len(i) for i in dataset]), 6)
        except ZeroDivisionError:
            net_results["purchase occurrence"] = 0
        
        print(f"{BUY_TRIGGER} trigger : {net_results}")

    # save the model as a .pkl file
    pyt.save(model, path.join(NETS_DIR, f"TradingNet{net_ind}_2.pkl"))

    # print
    print(f"Model {net_ind + 1}/{len(table_lists)} complete after {datetime.now() - model_start_time}\n")

    # tick model number
    net_ind += 1
