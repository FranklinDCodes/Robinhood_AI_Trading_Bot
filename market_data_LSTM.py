# ---------------------------------- IMPORTS ----------------------------------
import datetime as dt
from sql_tools import db
from random import randint
from torch import tensor, float32
from numpy import random
import yfinance as yf
import datetime as dt
from pandas import DataFrame
# -----------------------------------------------------------



# ---------------------------------- CONSTANTS ----------------------------------
# rounding precision
ROUNDING = 8
# database name
DB = None
# set DB info if there is one
if DB is not None:
    # database object
    DBO = db(DB)
    # name of table containing stock data
    DATA_TABLE = list(DBO.table.keys())[0]
    # length of table containing stock data
    DATA_TABLE_LEN = len(DBO.table[DATA_TABLE].getAll())
else:
    # database object
    DBO = None
    # name of table containing stock data
    DATA_TABLE = None
    # length of table containing stock data
    DATA_TABLE_LEN = None
# number of candles to use as input for the LSTM
INPUT_CANDLE_COUNT = 200
# minutes being predicted by output of the LSTM
OUTPUT_MINUTES = 1440*2
# interval of the candles 
DATA_INTERVAL_MINS = 1440
# the number of candles being predicted by the LSTM output
OUTPUT_CANDLES = int(OUTPUT_MINUTES/DATA_INTERVAL_MINS)
# standard datetime format
DT_FORMAT = "%Y/%m/%d %H:%M:%S"
# length of the volume ema indicator
VOLUME_EMA_LEN = 14
# add gaussian noise to input
ADD_NOISE = True
# number of noise values per candlestick
NOISE_COUNT = 1
# add IDs for each stock
ADD_ID = True
# ID for stock input
ID = 0
# list of days of the week
day_list = ["Sun", "Mon", "Tue", "Wed", "Thur", "Fri", "Sat"]
# ids for each weekday
WEEKDAYS = {i[0]:round((i[1] + 1)/(7/2), ROUNDING) for i in zip(day_list, range(7))}
# -----------------------------------------------------------



# ---------------------------------- INDICATOR FUNCTIONS ----------------------------------
# calculates the money flow index
def moneyFlowIndex(dataSet, length = 14):

    candle_len = len(dataSet[0])

    prev = 0
    for i in dataSet:
        typicalPrice = (float(i[2]) + float(i[3]) + float(i[4]))/3
        rawFlow = typicalPrice*float(i[5])
        if typicalPrice > prev:
            isUp = True
        else:
            isUp = False
        prev = typicalPrice
        i.append([rawFlow, isUp])
    ind = 0
    goods = 1
    for i in dataSet:
        if ind < length:
            ind += 1
            i.append(None)
            continue
        else:
            goods += 1
            totalUp = 0
            totalDown = 0
            for idx in range(ind - length + 1, ind + 1):
                if dataSet[idx][candle_len][1]:
                    totalUp += dataSet[idx][candle_len][0]
                else:
                    totalDown += dataSet[idx][candle_len][0]
            try:
                mfr = totalUp/totalDown
            except ZeroDivisionError:
                mfr = 0
            mfi = 100-(100/(mfr+1))
            i.append(mfi)
            ind += 1
    for i in dataSet:
        i.pop(-2)
    prev = 0

    return dataSet

# calculates the exponential moving average
def EMA(dataSet, periods = 200, based_on_index = 4):
    
    mult = 2/(periods + 1)
    mult2 = 1 - mult
    final = []
    ind = 0
    pastVals = [float(i[based_on_index]) for i in dataSet[0:periods]]
    lastLen = 0
    for i in dataSet:
        if ind < periods - 1:
            ind += 1
            continue
        elif ind == periods - 1:
            sma = sum(pastVals)/periods
            i.append(sma)
            i.append('SMA')
            final.append(i)
            lastLen = len(i) - 1
            ind += 1
        else:
            pastVals.pop(0)
            pastVals.append(float(i[based_on_index]))
            EMA = float(i[based_on_index]) * mult + dataSet[ind - 1][lastLen - 1] * mult2
            i.append(EMA)
            final.append(i)
            lastLen = len(i)
            ind += 1

    return dataSet

# calculates candlestick pattern indicators
def candle_indicators(dataset):

    ind = 0
    for cand in dataset:
        open = cand[1]
        high = cand[2]
        low = cand[3]
        close = cand[4]
        engulfing = 0
        hammer_or_star = 0
        if ind != 0:
            prev_open = dataset[ind - 1][1]
            prev_close = dataset[ind - 1][4]
            if open < close and prev_open > prev_close and open < prev_close and close > prev_open:
                engulfing = 1
            elif open > close and prev_open < prev_close and open > prev_close and close < prev_open:
                engulfing = -1
        if open < close:
            if (open - low) >= (close - open)*2 and (high - close) < (close - open):
                hammer_or_star = 1
        elif open > close:
            if (close - low) >= (open - close)*2 and (high - open) < (open - close):
                hammer_or_star = -1
        dataset[ind].append(engulfing)
        dataset[ind].append(hammer_or_star)
        ind += 1

    return dataset

# adds all the indicators to the dataset
def getIndicators(dataSet):

    # add mfi to dataset
    mfi_added = moneyFlowIndex(dataSet)

    # add 200 ema to dataset
    ma_added = EMA(mfi_added, 200)

    # add candlestick pattern indicators
    candle_patterns_added = candle_indicators(ma_added)

    # add volume ema
    final = EMA(candle_patterns_added, VOLUME_EMA_LEN, 5)

    # remove extra data
    try:
        for i in range(200):
            final.pop(0)
    except IndexError:
        return None

    return final
# -----------------------------------------------------------



# ---------------------------------- DATA FUNCTIONS ----------------------------------
# generate historical dataset
def getHistoricalData(indexes = None):

    # gather data in either a subset or all of it
    if indexes is None:
        data = DBO.table[DATA_TABLE].getAll()
    else:
        data = DBO.sql(f"SELECT * FROM {DATA_TABLE} WHERE ind >= {indexes[0]} AND ind <= {indexes[1]};")
    
    # add candle size and set time as minutes of the day
    data_2 = []
    for ind, tm, opn, hg, lw, clse, vol in data:
        time = dt.datetime.strptime(tm, DT_FORMAT)
        data_2.append([ind, time, opn, hg, lw, clse, vol])
    
    # unzip indexes
    data_2_indexs = [i[0] for i in data_2]
    data_2 = [[i[1], i[2], i[3], i[4], i[5], i[6]] for i in data_2]

    # calculate indexes
    data_indicators = getIndicators(data_2)

    # check to make sure there is enough data
    if data_indicators is None:
        return None

    # re-zip indexes
    data_return = []
    for i in range(200):
        data_2_indexs.pop(0)
    ind = 0
    for i in data_indicators:
        add = i
        add.insert(0, data_2_indexs[ind])
        data_return.append(add)
        ind += 1

    return data_return

# download live yahoo data
def downloadData(ticker):
    
    # get data
    try:
        df = yf.download(ticker, (dt.datetime.now() - dt.timedelta(days=200*1.4 + 50 + INPUT_CANDLE_COUNT*1.4)).strftime("%Y-%m-%d"), dt.datetime.now().strftime("%Y-%m-%d"), progress=False)
    except:
        return None

    # fix data col
    df = df.reset_index()
    df["Date"] = [dt.datetime.strptime(str(i), "%Y-%m-%d 00:00:00").strftime("%Y/%m/%d 00:00:00") for i in df['Date']]

    # delete col
    del df["Adj Close"]

    # rename columns
    df.columns = ["close_time", "open", "high", "low", "close", "volume"]

    # insert index
    df.insert(0, "ind", list(range(len(df))))

    # return data
    return df.iloc

# generate a candlestick for today thus far
def getCurrentCandle(ticker):
    
    # get data
    try:
        df = yf.download(ticker, (dt.datetime.now() - dt.timedelta(hours=24)), dt.datetime.now(), progress=False, interval="5m")
    except:
        return None

    if len(df) < 50:
        return None

    # fix date col
    df = df.reset_index()
    df["Datetime"] = [dt.datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S").strftime("%Y/%m/%d") for i in df['Datetime']]

    # delete adj close col
    del df["Adj Close"]

    # rename columns
    df.columns = ["close_time", "open", "high", "low", "close", "volume"]

    # insert index
    df.insert(0, "ind", list(range(len(df))))

    # remove yesterdays data 
    df = df[df["close_time"] == dt.datetime.now().strftime("%Y/%m/%d")]

    # get candle prices
    timestamp = dt.datetime.now().strftime("%Y/%m/%d 00:00:00")
    open_price = df.iloc[0]["open"]
    high_price = max(df["high"])
    low_price = min(df["low"])
    close_price = df.iloc[-1]["close"]
    volume = sum(df["volume"])

    # return data
    return [timestamp, open_price, high_price, low_price, close_price, volume]

# generate live stock data
def getLiveData(ticker, use_today_thus_far=True):

    # gather data
    data = list(downloadData(ticker))

    # check for successful download
    if len(data) == 0:
        return None

    # add todays candle
    if use_today_thus_far:
        todays_candle = getCurrentCandle(ticker)

        if todays_candle is None:
            return None
        
        # add index to candle
        last_index = data[-1][0]
        todays_candle.insert(0, last_index)

        # add it to dataset
        data.append(todays_candle)

    # add candle size and set time as minutes of the day
    data_2 = []
    for ind, tm, opn, hg, lw, clse, vol in data:
        time = dt.datetime.strptime(tm, DT_FORMAT)
        data_2.append([ind, time, opn, hg, lw, clse, vol])
    
    # unzip indexes
    data_2_indexs = [i[0] for i in data_2]
    data_2 = [[i[1], i[2], i[3], i[4], i[5], i[6]] for i in data_2]

    # calculate indexes
    data_indicators = getIndicators(data_2)

    # check to make sure there is enough data
    if data_indicators is None:
        return None

    # re-zip indexes
    data_return = []
    for i in range(200):
        data_2_indexs.pop(0)
    ind = 0
    for i in data_indicators:
        add = i
        add.insert(0, data_2_indexs[ind])
        data_return.append(add)
        ind += 1

    return data_return

# make dataset relative
def makeRelative(dataSet):

    # setup values
    global ROUNDING
    price1 = 1
    vol1 = 0
    final = []
    index = 0

    # open dataset
    for ind, stamp, open, high, low, close, vol, mfi, ema, engulf, hammer, vol_ema in dataSet:

        # set starting values for relative calculations
        if index == 0:
            price1 = open
            if vol != 0:
                vol1 = vol

        # check if starting volume value can be update to a non-0 value
        elif vol1 == 0:
            vol1 = vol

        # set volume if vol1 == 0 to avoid zero division error
        if vol1 == 0:
            vol = 0
        else:
            vol = round(vol/vol1, ROUNDING)

        # set volume ema
        if vol1 != 0:
            vol_ema = round(vol_ema/vol1, ROUNDING)
        else:
            vol_ema = 0.0
        
        # set time values
        dt_year_pos = int(stamp.strftime("%j"))/(365*.5)
        dt_week_pos = (int(stamp.strftime("%w")) + 1)/(7*.5)

        # make candle relative
        add = [ID, round(dt_year_pos, ROUNDING), round(dt_week_pos, ROUNDING), round(open/price1, ROUNDING), round(high/price1, ROUNDING), round(low/price1, ROUNDING), 
                        round(close/price1, ROUNDING), float(vol), round(mfi/100, ROUNDING), round(ema/price1, ROUNDING), float(engulf), float(hammer), vol_ema]                

        # add noise
        for i in range(NOISE_COUNT):
            add.insert(0, random.normal(0, 1))

        # add candle
        final.append(add)
        index += 1

    return final
# -----------------------------------------------------------



# ---------------------------------- USER FUNCTIONS ----------------------------------
# pull and format live data for network input
def getLiveInput(stock_ticker, use_today_thus_far=True):
    global INPUT_CANDLE_COUNT
    global ID

    # get data
    hist = getLiveData(stock_ticker, use_today_thus_far)

    # check to make sure there is enough data
    if hist is None:
        return None

    # create final input
    dataSet = makeRelative(hist[-INPUT_CANDLE_COUNT:])

    # format and return
    return tensor(dataSet, dtype=float32)

# generate training example
def getTrainingExample(index="random", return_est_low=False, profit_aim = 0.01, stop_loss=-0.01):
    global OUTPUT_MINUTES
    global INPUT_CANDLE_COUNT

    # pick random index for simulated moment, or use arg
    if index == "random":
        specific_index = randint(INPUT_CANDLE_COUNT + 200 + 1, DATA_TABLE_LEN - (int(OUTPUT_MINUTES/DATA_INTERVAL_MINS) + 2))
    else:
        if index > len(list(range(INPUT_CANDLE_COUNT + 200 + 1, DATA_TABLE_LEN - (int(OUTPUT_MINUTES/DATA_INTERVAL_MINS) + 2)))) - 1:
            return 0
        specific_index = list(range(INPUT_CANDLE_COUNT + 200 + 1, DATA_TABLE_LEN - (int(OUTPUT_MINUTES/DATA_INTERVAL_MINS) + 2)))[index]

    # get data
    hist = getHistoricalData((specific_index - (INPUT_CANDLE_COUNT + 220), specific_index + 2))

    # check to make sure there is enough data
    if hist is None:
        return None

    # change the indexes to corrospond to subset of data
    ind_count = 0
    for i in hist:
        ind = i[0]
        if ind == specific_index:
            index_new = ind_count
            break
        ind_count += 1

    # create final input
    dataSet = makeRelative(hist[index_new - INPUT_CANDLE_COUNT: index_new])
    
    # calculate data labels
    increases = []
    for y in range(INPUT_CANDLE_COUNT):
        
        # important prices
        last_cand = index_new - INPUT_CANDLE_COUNT + y
        highPrice = max([i[3] for i in hist[last_cand + 1:last_cand + 1 + OUTPUT_CANDLES]])
        closePrice = hist[last_cand][5]
        increase = (highPrice - closePrice)/closePrice
        lastPrice = hist[last_cand + 1:last_cand + 1 + OUTPUT_CANDLES][-1][5]
        decrease = (lastPrice - closePrice)/closePrice

        # create label
        if increase >= profit_aim:
            if stop_loss is not None and decrease > stop_loss:
                increases.append([1.0])
            elif stop_loss is not None and decrease <= stop_loss:
                increases.append([0.0])
            else:
                increases.append([1.0])
        else:
            increases.append([0.0])

    # return
    if return_est_low:
        return tensor(dataSet, dtype=float32), tensor(increases, dtype=float32), decrease
    else:
        return tensor(dataSet, dtype=float32), tensor(increases, dtype=float32)

# get the number of possible training examples from dataset
def getTrainingExampleCount():
    return len(range(INPUT_CANDLE_COUNT + 200 + 1, DATA_TABLE_LEN - (int(OUTPUT_MINUTES/DATA_INTERVAL_MINS) + 2)))
# -----------------------------------------------------------
