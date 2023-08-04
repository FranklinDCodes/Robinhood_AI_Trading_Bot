import market_data_LSTM as md
import torch as pyt
from torch import nn
from os import path, listdir
from datetime import datetime as dt

# original nets
class MARKET_LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = pyt.nn.Sequential(
            nn.Linear(128, 1)
            )
        self.input_is_binary = True
        self.profit_aim = 0.01
        self.stop_loss = None
        self.input_candles = 200
        self.output_minutes = None # (1440 or 1440*2)
        self.stock_list = []

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

# list of stocks to leave out of the market report
BLACKLISTED_TICKERS = []

def get_market_report():

    global MARKET_LSTM

    # forward pass results 
    results = []

    # candle settings
    md.INPUT_CANDLE_COUNT = 200
    USE_LIVE_DATA = True

    # define directory
    directory = "nets"

    # set time
    start = dt.now()

    # open list of networks
    for model_file in listdir(directory):

        # load network
        model = pyt.load(path.join(directory, model_file))
        
        # network stocks
        ticker_ind = 0
        for ticker in model.stock_list:

            # check if ticker is blacklisted
            if ticker in BLACKLISTED_TICKERS:
                continue

            # set ID
            md.ID = ticker_ind/len(model.stock_list)

            # gather data
            input = md.getLiveInput(ticker, USE_LIVE_DATA)

            # check data
            if input is None:
                print(f"!!! NOT ENOUGH DATA FOR {ticker} !!!")
                ticker_ind += 1
                continue

            # forward pass 
            out = model(input)[-1]

            # add information to results list
            results.append([float(out[0]), ticker])

            # count stocks
            ticker_ind += 1

    # sort list
    results.sort(reverse=True)
    
    return results


