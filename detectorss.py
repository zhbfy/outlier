from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import matplotlib.pylab as plt
import datetime

# def gaussiaonSameHour(df, ts_start, win):
#     ts_start_dt = pd.to_datetime(ts_start, unit='s')
#     ts_start_dt_end = pd.to_datetime(ts_start + 60 * 60, unit='s')
#     if ts_start_dt in df.index:
#         temp_df = df.loc[ts_start_dt:ts_start_dt_end, ['Value']]
#     else:
#         temp_df = df.loc[ts_start_dt:, ['Value']]
#
#     ts_start = ts_start - win * 60
#     while ts_start in df.index:
#         ts_start_dt = pd.to_datetime(ts_start, unit='s')
#         ts_start_dt_end = pd.to_datetime(ts_start + 60 * 60, unit='s')
#         temp_df.append(df.loc[ts_start_dt:ts_start_dt_end, ['Value']])
#         ts_start = ts_start - win * 60
#
#     ts_start_dt = pd.to_datetime(ts_start, unit='s')
#     ts_start_dt_end = pd.to_datetime(ts_start + 60 * 60, unit='s')
#     if ts_start_dt not in df.index and ts_start_dt_end in df.index:
#         temp_df.append(df.loc[:ts_start_dt_end, ['Value']])
#
#     # print(temp_df)
#
#     return temp_df.mean(), temp_df.std()

def gaussiaonSameHour(df, residual, ts_start, win):
    ts_start = ts_start - win * 60
    while ts_start - win * 60 in df.index:
        ts_start = ts_start - win * 60

    ts_end = ts_start + 60 * 60
    ts_start_dt = pd.to_datetime(ts_start, unit='s')
    ts_end_dt = pd.to_datetime(ts_end, unit='s')
    # print(ts_start_dt)
    # print(ts_end_dt)

    temp_df = residual.loc[ts_start_dt:ts_end_dt, ['Value']]
    # print(temp_df)
    ts_start = ts_start + win * 60
    ts_end = ts_start + 60 * 60

    # temp_df = pd.concat([temp_df, residual.loc[ts_start_dt:ts_end_dt, ['Value']]])

    while ts_end < int(df.index[-1]):
        ts_start_dt = pd.to_datetime(ts_start, unit='s')
        ts_end_dt = pd.to_datetime(ts_end, unit='s')
        temp_df = pd.concat([temp_df, residual.loc[ts_start_dt:ts_end_dt, ['Value']]])
        # print(ts_start_dt)
        # print(ts_end_dt)
        #     print(residual.loc[ts_start_dt:ts_end_dt, ['Value']])
        ts_start = ts_start + win * 60
        ts_end = ts_start + 60 * 60
        # print(temp_df)

    ts_start_dt = pd.to_datetime(ts_start, unit='s')
    ts_end_dt = pd.to_datetime(ts_start + 60 * 60, unit='s')
    if ts_start_dt in df.index and ts_end_dt not in df.index:
        temp_df = pd.concat([temp_df, residual.loc[ts_start_dt:ts_end_dt, ['Value']]])

    # print(temp_df)

    mean = temp_df.mean()
    std = temp_df.std()
    return mean, std

def gaussiaonSameHourMad(df, residual, ts_start, win):
    ts_start = ts_start - win * 60
    while ts_start - win * 60 in df.index:
        ts_start = ts_start - win * 60

    ts_end = ts_start + 60 * 60
    ts_start_dt = pd.to_datetime(ts_start, unit='s')
    ts_end_dt = pd.to_datetime(ts_end, unit='s')
    # print(ts_start_dt)
    # print(ts_end_dt)

    temp_df = residual.loc[ts_start_dt:ts_end_dt, ['Value']]
    # print(temp_df)
    ts_start = ts_start + win * 60
    ts_end = ts_start + 60 * 60

    # temp_df = pd.concat([temp_df, residual.loc[ts_start_dt:ts_end_dt, ['Value']]])

    while ts_end < int(df.index[-1]):
        ts_start_dt = pd.to_datetime(ts_start, unit='s')
        ts_end_dt = pd.to_datetime(ts_end, unit='s')
        temp_df = pd.concat([temp_df, residual.loc[ts_start_dt:ts_end_dt, ['Value']]])
        # print(ts_start_dt)
        # print(ts_end_dt)
        #     print(residual.loc[ts_start_dt:ts_end_dt, ['Value']])
        ts_start = ts_start + win * 60
        ts_end = ts_start + 60 * 60
        # print(temp_df)

    ts_start_dt = pd.to_datetime(ts_start, unit='s')
    ts_end_dt = pd.to_datetime(ts_start + 60 * 60, unit='s')
    if ts_start_dt in df.index and ts_end_dt not in df.index:
        temp_df = pd.concat([temp_df, residual.loc[ts_start_dt:ts_end_dt, ['Value']]])

    # print(temp_df)

    mean = temp_df.mean()
    # std = temp_df.std()
    median = float(temp_df.median())
    mad = temp_df["Value"].apply(lambda x: np.abs(x - median)).median()
    return mean, mad

def test_stationarity(timeseries):  # 时间序列平稳行检验
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# ---------------------------------下面是detector函数--------------------------------------------------


def calculateDiff(df, timestamp, parameter):
    # if parameter == 'last-slot':
    #     interval = 1 * 60
    # elif parameter == 'last-day':
    #     interval = 24 * 60 * 60
    # elif parameter == 'last-week':
    #     interval = 7 * 24 * 60 * 60
    interval = parameter * 60

    temp_ts = timestamp - interval
    try:
        diff_value = df.loc[timestamp, 'Value'] - df.loc[temp_ts, 'Value']
    except KeyError:
        raise Exception('invalid timestamp:', timestamp)
    else:
        return diff_value



def calculateEWMA(df, timestamp, alpha):
    new_df = df.loc[:timestamp, ['Value']]
    return new_df.ewm(alpha=alpha, adjust=False).mean().loc[timestamp, 'Value']



def calculateARIMA(df, timestamp):  # 有问题 数值取的太靠前会报错
    # old_timestamp = timestamp
    # new_df = df.loc[:timestamp, ['Value']]
    timestamp = pd.to_datetime(timestamp, unit='s')
    ts = df['Value']
    ts.index = pd.to_datetime(ts.index, unit='s')
    # ts_log = np.log(ts)
    arma_model = ARIMA(ts, order=(3, 0, 2)).fit(disp=-1)  # 好像目前的数据集已经满足adf检验 所以就没有做差分  //3,5?
    # timestamp = pd.to_datetime(timestamp, unit='s')
    predict_data = float(arma_model.predict(start=timestamp, end=timestamp, dynamic=False))
    return predict_data



def calculateTSD(df, timestamp, win):
    #  win 是参数，即时间窗口是多少分钟，例如1天就是24 * 60、一周7*24*60
    decompfreq = win
    ts = df[['Value']]
    ts.index = pd.to_datetime(ts.index, unit='s')

    decomposition = seasonal_decompose(ts, freq=decompfreq, model='additive', two_sided=True)
    residual = decomposition.resid
    # plt.plot(residual)
    # plt.savefig('residual.png')
    # residual.dropna(inplace=True)
    # print(residual)
    # print(residual[residual["Value"]!=np.NaN])

    query_ts = pd.to_datetime(timestamp, unit='s')
    if np.isnan(residual.loc[query_ts, 'Value']):
        raise Exception('invalid timestamp:', timestamp)

    # print('query_ts')
    # print(query_ts)
    q_y = query_ts.year
    q_m = query_ts.month
    q_d = query_ts.day
    q_h = query_ts.hour
    # hour_start = int(time.mktime((q_y, q_m, q_d, q_h, 0, 0, 0, 0, 0)))
    # hour_end = pd.to_datetime(datetime(q_y, q_m, q_d, q_h + 1))
    tz_utc = datetime.timezone(datetime.timedelta(hours=0))
    ts_start = int(datetime.datetime(q_y, q_m, q_d, q_h, 0, 0, tzinfo=tz_utc).timestamp())
    # print('ts_start')
    # print(datetime.datetime(q_y, q_m, q_d, q_h, 0, 0, tzinfo=tz_utc))
    mean, std = gaussiaonSameHour(df, residual, ts_start, win)
    # print(mean)
    # print(std)

    return float(np.abs((residual.loc[query_ts, 'Value'] - mean))/std)

def calculateTSDmad(df, timestamp, win):
    #  win 是参数，即时间窗口是多少分钟，例如1天就是24 * 60、一周7*24*60
    decompfreq = win
    ts = df[['Value']]
    ts.index = pd.to_datetime(ts.index, unit='s')

    decomposition = seasonal_decompose(ts, freq=decompfreq, model='additive', two_sided=True)
    residual = decomposition.resid
    # plt.plot(residual)
    # plt.savefig('residual.png')
    # residual.dropna(inplace=True)
    # print(residual)
    # print(residual[residual["Value"]!=np.NaN])

    query_ts = pd.to_datetime(timestamp, unit='s')
    if np.isnan(residual.loc[query_ts, 'Value']):
        raise Exception('invalid timestamp:', timestamp)

    # print('query_ts')
    # print(query_ts)
    q_y = query_ts.year
    q_m = query_ts.month
    q_d = query_ts.day
    q_h = query_ts.hour
    # hour_start = int(time.mktime((q_y, q_m, q_d, q_h, 0, 0, 0, 0, 0)))
    # hour_end = pd.to_datetime(datetime(q_y, q_m, q_d, q_h + 1))
    tz_utc = datetime.timezone(datetime.timedelta(hours=0))
    ts_start = int(datetime.datetime(q_y, q_m, q_d, q_h, 0, 0, tzinfo=tz_utc).timestamp())
    # print('ts_start')
    # print(datetime.datetime(q_y, q_m, q_d, q_h, 0, 0, tzinfo=tz_utc))
    mean, mad = gaussiaonSameHourMad(df, residual, ts_start, win)
    # print(mean)
    # print(std)

    return float(np.abs((residual.loc[query_ts, 'Value'] - mean))/mad)
