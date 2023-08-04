# ⁡⁣⁢|-----------------{ IMPORTS }-----------------|
import sql_tools
from sql_tools import db
from csv import reader, writer
from datetime import datetime, timedelta
import time
from os import path
import daily_market_report as market
from robin_stocks import robinhood as r
import pyotp
from hashlib import sha256
import tkinter as tk
import requests
import torch as pyt
from torch import nn



# ⁡⁣⁢⁢⁡⁣⁢⁣⁡⁣⁢|-----------------{ SETTINGS }-----------------|

# the maximum decimal percent of the balance that can be in use at a given time
BUY_BUFFER = .95

# profit aim
WIN_AIM = 0.01

# percent of win aim to lose before stop loss
WIN_RISK = 0.08

# the minimum signal strength that will trigger a purchase
PURCHASE_INDICATOR_THRESHOLD = 0.7

# whether or not to only buy whole shares
USE_WHOLE_SHARES = False

# time to rest between scans in seconds
SLEEP_TIME = 120

# tickers that are ignored in market scanning
market.BLACKLISTED_TICKERS = []

# how often to print loop number
PRINT_LOOPS = 20



# |-----------------{ DATETIME }-----------------|

# standard date format
DATE_FORMAT = "%m/%d/%Y"

# standard date and time format
DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

# is robinhood normal trading hours
def is_market_hours():
    link_data = requests.get(r.get_markets()[1]['todays_hours']).text
    spot = link_data.index('is_open') + 9
    open_today = link_data[spot:spot + 4] == 'true'
    now = datetime.now()
    market_open = datetime.strptime(now.strftime(DATE_FORMAT) + " 08:30:00", DATETIME_FORMAT)
    market_close = datetime.strptime(now.strftime(DATE_FORMAT) + " 15:00:00", DATETIME_FORMAT)
    return now > market_open and now < market_close and open_today

# is after 14:15
def is_after_230():
    now = datetime.now()
    _230 = datetime.strptime(now.strftime(DATE_FORMAT) + " 14:15:00", DATETIME_FORMAT)
    return now > _230



# |-----------------{ DIRECTORIES }-----------------|
OP_DATA_DIR = "operating_data"
NETS_DIR = "nets"
STATS_DIR = "statistics"
LOG_DIR = "logs"



# ⁡⁣⁢|-----------------{ DATABASE FILES }-----------------|

# Logs
LOGS = db("logs.db", LOG_DIR)

# Setup tables
if len(LOGS.table) == 0:
    # Log of all movements of money (side = "b" or "s" or "deposit" or "withdrawal")
    LOGS.newTable("balance_log", ("timestamp", "side", "amount", "old_buying_power", "new_buying_power", "new_est_equity_net_worth", "system_id"))
    # Log of all opened and close positions (increase = profit %)
    LOGS.newTable("full_trade_log", ("purchase_timestamp", "sale_timestamp", "symbol", "purchase_price", "sale_price", "shares", "profit", "increase", "system_id"))
    # Log of all purchases and sales (id = full trade id + b for buy, + s for sale) (estimated_price == price hoped to purchase or sell at)
    LOGS.newTable("transaction_log", ("timestamp", "side", "symbol", "estimated_price", "actual_price", "shares", "system_id"))
    # Log of all money transfers (type = withdrawal or deposit)
    LOGS.newTable("transfer_log", ("timestamp", "type", "amount", "system_id"))

# INIT tables
BAL_LOG: sql_tools.db.Table = LOGS.table["balance_log"]
FULL_TRADE_LOG: sql_tools.db.Table = LOGS.table["full_trade_log"]
TRANSACTION_LOG: sql_tools.db.Table = LOGS.table["transaction_log"]
TRANSFER_LOG: sql_tools.db.Table = LOGS.table["transfer_log"]

# logs ids
ID_FILE = path.join(LOG_DIR, "id.txt")
def transfer_id():
    with open(ID_FILE, "r") as r:
        current = int(r.readline())
    with open(ID_FILE, "w") as r:
        r.writelines([str(current + 1)])
    return "#tfr" + str(current)
def trade_id():
    with open(ID_FILE, "r") as r:
        current = int(r.readline())
    with open(ID_FILE, "w") as r:
        r.writelines([str(current + 1)])
    return "#tde" + str(current)



# |-----------------{ CSV & TXT FILES }-----------------|

# [ticker, side, est_price, robinhood_id]
UNCONFIRMED_STOCKS_DIR = path.join(OP_DATA_DIR, "unconfirmed.csv")

# [timstamp, id, symb, price, shares, stock is profitable (passed profit aim), stop_loss (only if profitable to keep from losing win)]
OWNED_STOCKS_DIR = path.join(OP_DATA_DIR, "owned.csv")

# [datetime]
LAST_RUN_DIR = path.join(OP_DATA_DIR, "last_run.txt")



# |-----------------{ DATA TOOLS }-----------------|

# file handlers
def read(flname: str):
    with open(flname, "r") as fl:
        return list(fl.readlines())
def write(flname: str, values: list[any], add_line_return = True):
    with open(flname, "w") as fl:
        if add_line_return:
            values = [str(i) + '\n' for i in values]
        fl.writelines(values)

# csv file handlers⁡
def cread(flname: str):
    with open(flname, "r") as fl:
        return list(reader(fl.readlines()))
def cwrite(flname: str, values: list[list[any]]):
    #if not len(values) == 0:    
    with open(flname, "w") as fl:
        w = writer(fl, lineterminator="\n")
        w.writerows(values)
def cappend(flname: str, value: list[any]):
    current = cread(flname)
    current.append(value)
    cwrite(flname, current)

# find and log if purchase code has been run
def has_purchased_today():
    return datetime.strptime(read(LAST_RUN_DIR)[0], DATE_FORMAT).strftime(DATE_FORMAT) == datetime.now().strftime(DATE_FORMAT)
def set_purchased_today():
    write(LAST_RUN_DIR, [datetime.now().strftime(DATE_FORMAT)], False)

# confirm purchase that has been filled
def confirm_transaction(symb, side, est_price, actual_price, shares, robinhood_id):

    # if buy
    if side == "b":

        # new id
        sys_id = trade_id()

        transaction_time = datetime.now().strftime(DATETIME_FORMAT)

        # add to owned
        cappend(OWNED_STOCKS_DIR, [transaction_time, sys_id, symb, actual_price, shares, False, 0])

    # if sell
    else:

        # find in owned
        owned_records = cread(OWNED_STOCKS_DIR)
        owned_index = [i[2] for i in owned_records].index(symb)
        buy_time, sys_id, symb, buy_price, shares, is_profitable, stop_loss = owned_records[owned_index]
        buy_price, shares = float(buy_price), float(shares)
        
        # remove from owned
        owned_records.pop(owned_index)
        cwrite(OWNED_STOCKS_DIR, owned_records)

        transaction_time = datetime.now().strftime(DATETIME_FORMAT)

    #LOGS.newTable("transaction_log", ("timestamp", "side", "symbol", "estimated_price", "actual_price", "shares", "system_id"))
    TRANSACTION_LOG.insert([transaction_time, side, symb, est_price, actual_price, shares, sys_id + side])

    # remove transaction from unconfirmed transaction records
    uncon = cread(UNCONFIRMED_STOCKS_DIR)
    uncon.pop([i[3] for i in uncon].index(robinhood_id))
    cwrite(UNCONFIRMED_STOCKS_DIR, uncon)

    # full log if a sell
    if side == "s":

        # full trade log: "purchase_timestamp", "sale_timestamp", "symbol", "purchase_price", "sale_price", "shares", "profit", "increase", "id"
        FULL_TRADE_LOG.insert([buy_time, transaction_time, symb, buy_price, actual_price, shares, (actual_price - buy_price)*shares, (actual_price - buy_price)/buy_price, sys_id])

# update profitable statuses of owned stocks (whether they've crossed original goal)
def update_owned():

    new_owned = []

    # open owned
    for buy_time, sys_id, ticker, buy_price, shares, profitable, stop_loss in cread(OWNED_STOCKS_DIR):
        
        # if not market as being above profit aim
        if profitable == "False":

            # check if it is above now
            bid_price = live_price(ticker)
            aim_price = float(buy_price)*WIN_AIM + float(buy_price)
            if bid_price >= aim_price:

                # if so update
                new_stop_loss = bid_price - (float(buy_price)*WIN_AIM*WIN_RISK)
                new_owned.append([buy_time, sys_id, ticker, buy_price, shares, True, new_stop_loss])
                
            else:

                # add with no changes
                new_owned.append([buy_time, sys_id, ticker, buy_price, shares, profitable, stop_loss])

        # elif profitable is true
        else:
            
            # recalculate stop-loss
            bid_price = live_price(ticker)
            new_stop_loss = bid_price - (float(buy_price)*WIN_AIM*WIN_RISK)

            # if price is higher, set new stop-loss
            if new_stop_loss > float(stop_loss):
                new_owned.append([buy_time, sys_id, ticker, buy_price, shares, True, new_stop_loss])
            else:
                new_owned.append([buy_time, sys_id, ticker, buy_price, shares, True, stop_loss])
    
    # write new data
    cwrite(OWNED_STOCKS_DIR, new_owned)



# |-----------------{ ROBINHOOD FUNCTIONS }-----------------|

# pw files
def set_pw_file(new_pass):

    encrypted_lines = []

    for char in new_pass:

        encrypted_lines.append(sha256(char.encode("utf-8")).hexdigest())

    write(path.join(OP_DATA_DIR, "pw.txt"), encrypted_lines)

    return 0
def get_pw():

    # final pw
    pw = ''

    # possible characters
    str_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
     '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', 
     '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # open encrypted pw
    for line in read(path.join(OP_DATA_DIR, "pw.txt")):

        # remove new lines
        if line[-1] == "\n":
            line = line[:-1]
        
        # find char to match line
        for char in str_chars:
            if sha256(char.encode("utf-8")).hexdigest() == line:
                pw += char
                break
    
    return pw

# login
def robin_login():
    totp = pyotp.TOTP(read(path.join(OP_DATA_DIR, 'otp.txt'))[0]).now()
    robin_login = r.login("emil@email.com", get_pw(), mfa_code=totp, store_session=False)
    return robin_login

# get user buying power
def buying_power():
    info = r.profiles.load_account_profile()['buying_power']
    return float(info)

# takes in a list of [[signal, symb]...] and returns {signal: funds alloted...}
def split_funds(suggestions, funds, split_factor = 3):

    # limit suggestions
    suggestions = suggestions[:20]

    # results
    results = []

    # raise the signals to a power and sum
    signals_squared = sum([i[0]**split_factor for i in suggestions])

    # determine funds as the signal percent of above sum
    for signal, symb in suggestions:
        percent = signal**split_factor/signals_squared
        results.append([round(percent*funds, 2), symb])

    # make sure rounding didn't mess up the 100 sum
    results.sort()
    results_sum = sum([i[0] for i in results])
    if results_sum != funds:
        results[-1][0] += funds - results_sum

    # sort and convert to dict
    return {i[1]: round(i[0], 2) for i in results}

# get information of an order based on order id
def order_info(id):
    info = r.orders.get_stock_order_info(id)
    # [status, price, quantity] status is "filled" or not
    return [info['state'], float(info['price']), float(info['quantity'])]

# get the live price of a stock or stocks. returns the ask price by defaultd
def live_price(symb, bid_price = True):

    # set price type
    price_type = None
    if bid_price:
        price_type = "bid_price"
    else:
        price_type = "ask_price"

    # get prices
    prices = r.stocks.get_latest_price(symb, price_type)

    # return single
    return float(prices[0])

# returns true if there is an upcoming or recent earnings report
def nearby_earnings_report(symb):

    # get recent earnings
    report_dates = [i['report']['date'] for i in r.stocks.get_earnings(symb)]
    today = datetime.now()

    # days away that shouldn't have earnings calls based on the weekday 0 == Monday... 6 == Sunday
    no_buys = {0: (-3, -2, -1, 0, 1), 1: (-1, 0, 1), 2:(-1, 0, 1), 3:(-1, 0, 1), 4:(-1, 0, 1, 2, 3)}

    # open recent earnings dates
    for date in report_dates:

        # if earnings is nearby 
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        no_buy_dates = [(date_obj + timedelta(days = i)).strftime(DATE_FORMAT) for i in no_buys[date_obj.weekday()]]
        if today.strftime(DATE_FORMAT) in no_buy_dates:
            
            # true meaning do not purchase
            return True
    
    # false meaning good to purchase
    return False

# transaction functions
def purchase(ticker, dollar_amount):
    order = r.orders.order_buy_fractional_by_price(ticker, dollar_amount)
    return order['id']
def sell(ticker, shares):
    info = r.orders.order_sell_fractional_by_quantity(ticker, shares)
    return info['id']



# |-----------------{ NEURAL NET }-----------------|

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



# |-----------------{ GUI }-----------------|

# quit mechanism
run = True
def kill():
    global run
    run = False

# init window
window = tk.Tk()
window.title("QUIT")

# create button
kill_button = tk.Button(window, background="red", text="X", font=("Ariel", 180), command= kill)
kill_button.pack()

# update button
window.update()



# |-----------------{ MAIN LOOP }-----------------|

# open loop
loops = 0
while run:

    # update gui
    window.update()

    # login with error catcher
    login_successful = False
    while not login_successful:
        try:
            robin_login()
            login_successful = True
            if loops == 0:
                print(f'{datetime.now().strftime(DATETIME_FORMAT)} | Robinhood login successful.\n')
        except:
            print(f'\n{datetime.now().strftime(DATETIME_FORMAT)} | !Robinhood login unsuccessful. Please check your internet connection.\n')
            time.sleep(10)
    
    #   <----- PURCHASE ----->
    # check if purchases need to be made
    if not has_purchased_today() and is_market_hours() and is_after_230():

        # get list
        print(f"{datetime.now().strftime(DATETIME_FORMAT)} | AI Picking Purchases...")
        purchase_suggestions = [i for i in market.get_market_report() if i[0] > PURCHASE_INDICATOR_THRESHOLD]

        # check list
        revised_suggestions = []
        for signal, ticker in purchase_suggestions:

            # check for recent or upcoming earnings reports
            if nearby_earnings_report(ticker):
                print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} not purchased due to earnings report\n")
                continue

            # make sure stock is not currently owned or has any unconfirmed transactions
            if ticker in [i[0] for i in cread(UNCONFIRMED_STOCKS_DIR)]:
                print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} not purchased due to unconfirmed transaction\n")
                continue

            # make sure stock is not currently owned or has any unconfirmed transactions
            if ticker in [i[1] for i in cread(OWNED_STOCKS_DIR)]:
                print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} not purchased due to ownership\n")
                continue
            
            # add to suggestions if still good
            revised_suggestions.append([signal, ticker])
        
        # divide funds
        buy_list = split_funds(revised_suggestions, buying_power()*BUY_BUFFER, 2)

        # open list if not empty
        if len(buy_list) != 0:
            for ticker in buy_list:

                # make purchase
                robinhood_id = purchase(ticker, buy_list[ticker])
                est_price = live_price(ticker, False)
                print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} | purchase placed | est. total -${buy_list[ticker]}\n")

                # record unconfirmed purchase
                cappend(UNCONFIRMED_STOCKS_DIR, [ticker, "b", est_price, robinhood_id])
    
        # record that purchase took place today
        set_purchased_today()


    #   <----- SELL ----->
    # check for time out sales
    if is_market_hours() and is_after_230():

        
        # get list of owned stocks
        for buy_time, sys_id, ticker, price, shares, profitable, stop_loss in cread(OWNED_STOCKS_DIR):
            
            if datetime.strptime(buy_time, DATETIME_FORMAT).strftime(DATE_FORMAT) == datetime.now().strftime(DATE_FORMAT):
                continue

            # check for any unconfirmed transactions
            if ticker in [i[0] for i in cread(UNCONFIRMED_STOCKS_DIR)]:
                continue

            # make sales
            robinhood_id = sell(ticker, float(shares))
            est_price = live_price(ticker)
            print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} | sale placed | est. total ${float(shares)*est_price}")

            # record unconfirmed purchase
            cappend(UNCONFIRMED_STOCKS_DIR, [ticker, "s", est_price, robinhood_id])

    # check for success sales & stop loss ajustments
    if is_market_hours() and not is_after_230():

        # update profitable status of stocks & their stop-losses
        update_owned()

        for buy_time, sys_id, ticker, buy_price, shares, profitable, stop_loss in cread(OWNED_STOCKS_DIR):

            # if the price dips below stop-loss
            if live_price(ticker) <= float(stop_loss):

                # check for any unconfirmed transactions
                if ticker in [i[0] for i in cread(UNCONFIRMED_STOCKS_DIR)]:
                    continue

                # make sales
                robinhood_id = sell(ticker, float(shares))
                est_price = live_price(ticker)
                print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} | sale placed | est. total ${float(shares)*est_price}\n")

                # record unconfirmed purchases
                cappend(UNCONFIRMED_STOCKS_DIR, [ticker, "s", est_price, robinhood_id])            
    

    #   <----- CONFIRM TRANSACTIONS ----->
    for ticker, side, est_price, id in cread(UNCONFIRMED_STOCKS_DIR):

        # get the data on the placed orders that haven't been confirmed to have been filled
        state, price, quantity = order_info(id)

        # if the order was completed, confirm it
        if state == "filled":
            confirm_transaction(ticker, side, float(est_price), float(price), float(quantity), id)
            print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {ticker} | {['purchase', 'sale'][side == 's']} confirmed | {float(quantity)} shares at ${float(price)}\n")


    #   <----- SLEEP ----->
    # update gui
    window.update()
    
    # split up sleep
    for i in range(SLEEP_TIME*4):

        # sleep
        time.sleep(.25)

        # update window
        window.update()
        if not run:
            break
    
    # loop count
    loops += 1
    if not loops % PRINT_LOOPS:
        print(f"{datetime.now().strftime(DATETIME_FORMAT)} | {loops}")
        print()

