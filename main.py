import configparser
import pandas as pd
import numpy as np
import copy
import statistics as sta
import math
import random
import traceback

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import matplotlib.pylab as plt
import datetime
from sklearn.model_selection  import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier


def dataload(inputfile):
    #加载数据
    df = pd.read_csv(inputfile,sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train101-m.csv",sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train102.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train103.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train104.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train105.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train106.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train107.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train108.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train109.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train110.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train111.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train112.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train113.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train114.csv", sep=',')
    #df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train115.csv", sep=',')




    #df["Timestamp"] = df["Timestamp"].astype("int")
    #df["Label"]=df["Label"].astype("int")
    #df["Value"] = df["Value"].astype("float64")
    df_new = df.set_index("Timestamp") #原始训练数据集
    df_positive=df_new[df_new["Label"].isin(["1"])]
    ex_list = list(df_new["Label"])
    ex_list.remove(1)
    df_wholenegative=df_new[df_new["Label"].isin(ex_list)]
    df_negative = df_new[df_new["Label"].isin(ex_list)].sample(800)
    df_input=pd.concat([df_positive,df_negative]).sort_index(ascending=True)#采样后训练数据集
    return [df_new,df_positive,df_wholenegative,df_input]




def featuregenerate(df_new,df_positive,df_wholenegative):
    #生成特征数据
    conf = configparser.ConfigParser()
    conf.read("detectors\\DetectorsConfig")
    detectorsoption = [conf.get("detectors", "Simple threshold"),
                 conf.get("detectors", "Diff"),
                 conf.get("detectors", "Simple MA"),
                 conf.get("detectors", "Weighted MA"),
                 conf.get("detectors", "MA of diff"),
                 conf.get("detectors", "EWMA"),
                 conf.get("detectors", "TSD"),
                 conf.get("detectors", "TSD MAD"),
                 conf.get("detectors", "Historical average"),
                 conf.get("detectors", "Historical MAD"),
                 conf.get("detectors", "Holt-Winters"),
                 conf.get("detectors", "SVD"),
                 conf.get("detectors", "Wavelet"),
                 conf.get("detectors", "ARIMA")]


    feature_head =["Timestamp","Label"]#特征表头
    if detectorsoption[1]=="1":
        feature_head = feature_head + ["Diff_1_slot", "Diff_1_day", "Diff_1_week"]
    if detectorsoption[2]=="1":
        feature_head = feature_head+["Simple_MA_10","Simple_MA_20","Simple_MA_30","Simple_MA_40","Simple_MA_50"]
    if detectorsoption[3]=="1":
        feature_head = feature_head+["Weighted_MA_10","Weighted_MA_20","Weighted_MA_30","Weighted_MA_40","Weighted_MA_50"]
    if detectorsoption[4]=="1":
        feature_head = feature_head+["MA_of_diff_10","MA_of_diff_20","MA_of_diff_30","MA_of_diff_40","MA_of_diff_50"]
    if detectorsoption[5] == "1":
        feature_head = feature_head + ["EWMA_0.1", "EWMA_0.3", "EWMA_0.5", "EWMA_0.7","EWMA_0.9"]
    if detectorsoption[6] == "1":
        feature_head = feature_head + ["TSD_1", "TSD_2", "TSD_3", "TSD_4","TSD_5"]
    if detectorsoption[7] == "1":
        feature_head = feature_head + ["TSD_MAD_1", "TSD_MAD_2", "TSD_MAD_3", "TSD_MAD_4","TSD_MAD_5"]
    if detectorsoption[8]=="1":
        feature_head = feature_head +["Historical_average_1", "Historical_average_2", "Historical_average_3","Historical_average_4","Historical_average_5"]
    if detectorsoption[11] == "1":
        for i in range(1,6):
            for j in range(1,2):
                feature_head = feature_head + ["SVD_"+str(i*10)+"rows_"+str(2*j+1)+"weeks"]
        #feature_head.append("Simple_MA_10")
        #feature_head.append("Simple_MA_20")
        #feature_head.append("Simple_MA_30")
        #feature_head.append("Simple_MA_40")
        #feature_head.append("Simple_MA_50")
    feature = pd.DataFrame(columns=feature_head) #生成特征表
    print(feature)
    for index,row in df_positive.iterrows():#生成正例特征
        try:
            feature_insert = []
            index=int(index)
            feature_insert.append(index)
            feature_insert.append(df_new.loc[index,"Label"])
            #print(feature_insert)
            if detectorsoption[1]=="1": #Diff
                f11 = calculateDiff(df_new,index,1)
                f12 = calculateDiff(df_new,index, 1440)
                f13 = calculateDiff(df_new,index, 10080)
                feature_insert= feature_insert+[f11,f12,f13]
            if detectorsoption[2]=="1": #simple_MA
                f1 = simple_ma(df_new,index,10)
                f2 = simple_ma(df_new,index, 20)
                f3 = simple_ma(df_new,index, 30)
                f4 = simple_ma(df_new,index, 40)
                f5 = simple_ma(df_new,index, 50)
                feature_insert= feature_insert+[f1,f2,f3,f4,f5]
            if detectorsoption[3]=="1": #Weighted_MA
                f31 = weighted_ma(df_new,index,10)
                f32 = weighted_ma(df_new,index, 20)
                f33 = weighted_ma(df_new,index, 30)
                f34 = weighted_ma(df_new,index, 40)
                f35 = weighted_ma(df_new,index, 50)
                feature_insert= feature_insert+[f31,f32,f33,f34,f35]
            if detectorsoption[4]=="1": #MA of diff
                f41 = ma_diff(df_new,index,10)
                f42 = ma_diff(df_new,index, 20)
                f43 = ma_diff(df_new,index, 30)
                f44 = ma_diff(df_new,index, 40)
                f45 = ma_diff(df_new,index, 50)
                feature_insert= feature_insert+[f41,f42,f43,f44,f45]
            if detectorsoption[5] == "1":  # EWMA
                f51 = calculateEWMA(df_new, index, 0.1)
                f52 = calculateEWMA(df_new, index, 0.3)
                f53 = calculateEWMA(df_new, index, 0.5)
                f54 = calculateEWMA(df_new, index, 0.7)
                f55 = calculateEWMA(df_new, index, 0.9)
                feature_insert = feature_insert + [f51, f52, f53, f54, f55]
            if detectorsoption[6] == "1":  # TSD
                f61 = calculateTSD(df_new,index , 7*24)
                f62 = calculateTSD(df_new,index , 2*7*24)
                f63 = calculateTSD(df_new,index , 3*7*24)
                f64 = calculateTSD(df_new,index , 4*7*24)
                f65 = calculateTSD(df_new,index , 5*7*24)
                feature_insert = feature_insert + [f61, f62, f63, f64, f65]
            if detectorsoption[7] == "1":  # TSD MAD
                f71 = calculateTSDmad(df_new,index , 7*24)
                f72 = calculateTSDmad(df_new,index , 2*7*24)
                f73 = calculateTSDmad(df_new,index , 3*7*24)
                f74 = calculateTSDmad(df_new,index , 4*7*24)
                f75 = calculateTSDmad(df_new,index , 5*7*24)
                feature_insert = feature_insert + [f71, f72, f73, f74, f75]
            if detectorsoption[8] == "1":  # Historical averag
                f81 = historical_average(df_new, index, 1)
                f82 = historical_average(df_new, index, 2)
                f83 = historical_average(df_new, index, 3)
                f84 = historical_average(df_new, index, 4)
                f85 = historical_average(df_new, index, 5)
                feature_insert = feature_insert + [f81, f82, f83, f84, f85]
            if detectorsoption[11] == "1":  # svd
                for i in range(1, 6):
                    for j in range(1, 2):
                        feature_insert = feature_insert + [svd(df_new, index, i*10, 2*j+1)]
            #a = feature_insert.index("None")
                #feature_insert.append(f1)
                #feature_insert.append(f2)
                #feature_insert.append(f3)
                #feature_insert.append(f4)
                #feature_insert.append(f5)
        except Exception as e:
            pass
            #print('Exception: ', e)
        else:
            insertRow = pd.DataFrame([feature_insert], columns=feature_head)
            #print(insertRow)
            feature=feature.append(insertRow,ignore_index=True)
            #print(feature)
    trainingdata_num = copy.deepcopy(feature.iloc[:, 0].size)*2 #训练样本条数
    while feature.iloc[:,0].size<trainingdata_num:#生成负例特征
        try:
            index= df_wholenegative.sample(1).index[0]#int64index[0]=int
            #print(index)
            feature_insert = []
            feature_insert.append(index)
            feature_insert.append(df_new.loc[index, "Label"])
            #print(feature_insert)
            if detectorsoption[1]=="1": #Diff
                f11 = calculateDiff(df_new,index,1)
                f12 = calculateDiff(df_new,index, 1440)
                f13 = calculateDiff(df_new,index, 10080)
                feature_insert= feature_insert+[f11,f12,f13]
            if detectorsoption[2]=="1":
                f1 = simple_ma(df_new,index,10)
                f2 = simple_ma(df_new, index, 20)
                f3 = simple_ma(df_new, index, 30)
                f4 = simple_ma(df_new, index, 40)
                f5 = simple_ma(df_new, index, 50)
                feature_insert = feature_insert + [f1, f2, f3, f4, f5]
            if detectorsoption[3]=="1": #Weighted_MA
                f31 = weighted_ma(df_new,index,10)
                f32 = weighted_ma(df_new,index, 20)
                f33 = weighted_ma(df_new,index, 30)
                f34 = weighted_ma(df_new,index, 40)
                f35 = weighted_ma(df_new,index, 50)
                feature_insert= feature_insert+[f31,f32,f33,f34,f35]
            if detectorsoption[4]=="1": #MA of diff
                f41 = ma_diff(df_new,index,10)
                f42 = ma_diff(df_new,index, 20)
                f43 = ma_diff(df_new,index, 30)
                f44 = ma_diff(df_new,index, 40)
                f45 = ma_diff(df_new,index, 50)
                feature_insert= feature_insert+[f41,f42,f43,f44,f45]
            if detectorsoption[5] == "1":  # EWMA
                f51 = calculateEWMA(df_new, index, 0.1)
                f52 = calculateEWMA(df_new, index, 0.3)
                f53 = calculateEWMA(df_new, index, 0.5)
                f54 = calculateEWMA(df_new, index, 0.7)
                f55 = calculateEWMA(df_new, index, 0.9)
                feature_insert = feature_insert + [f51, f52, f53, f54, f55]
            if detectorsoption[6] == "1":  # TSD
                f61 = calculateTSD(df_new,index , 7*24)
                f62 = calculateTSD(df_new,index , 2*7*24)
                f63 = calculateTSD(df_new,index , 3*7*24)
                f64 = calculateTSD(df_new,index , 4*7*24)
                f65 = calculateTSD(df_new,index , 5*7*24)
                feature_insert = feature_insert + [f61, f62, f63, f64, f65]
            if detectorsoption[7] == "1":  # TSD MAD
                f71 = calculateTSDmad(df_new,index , 7*24)
                f72 = calculateTSDmad(df_new,index , 2*7*24)
                f73 = calculateTSDmad(df_new,index , 3*7*24)
                f74 = calculateTSDmad(df_new,index , 4*7*24)
                f75 = calculateTSDmad(df_new,index , 5*7*24)
                feature_insert = feature_insert + [f71, f72, f73, f74, f75]
            if detectorsoption[8] == "1":  # Historical averag
                f81 = historical_average(df_new, index, 1)
                f82 = historical_average(df_new, index, 2)
                f83 = historical_average(df_new, index, 3)
                f84 = historical_average(df_new, index, 4)
                f85 = historical_average(df_new, index, 5)
                feature_insert = feature_insert + [f81, f82, f83, f84, f85]
            if detectorsoption[11] == "1":  # svd
                for i in range(1, 6):
                    for j in range(1, 2):
                        feature_insert = feature_insert + [svd(df_new, index, i * 10, 2 * j + 1)]

                #print(feature_insert)
                #feature_insert.append(f1)
                #feature_insert.append(f2)
                #feature_insert.append(f3)
                #feature_insert.append(f4)
                #feature_insert.append(f5)
        except Exception as e:
            #print('Exception: ', e)
            pass
        else:

            insertRow = pd.DataFrame([feature_insert], columns=feature_head)
            #print(insertRow)
            feature=feature.append(insertRow,ignore_index=True)
            #print(feature)

    feature = feature.set_index("Timestamp")
    feature.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    print(feature)

    return [detectorsoption,feature,feature_head]


def RFtraining(df):
    #随机森林
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    df.head
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    features = df.columns[:1]
    target = df.columns[1]
    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train[features], train[target])
    return clf

def RFCrossValidation(detectorsoption,df,inputfile):
    #带交叉验证的随机森林
    #feature_cols = ['Value']
    #feature_cols=copy.deepcopy(feature_head).remove('Timestamp')
    try:
        feature_cols=[]
        if detectorsoption[1]=="1":
            feature_cols = feature_cols +["Diff_1_slot", "Diff_1_day", "Diff_1_week"]
        if detectorsoption[2]=="1":
            feature_cols = feature_cols +["Simple_MA_10","Simple_MA_20","Simple_MA_30","Simple_MA_40","Simple_MA_50"]
        if detectorsoption[3] == "1":
            feature_cols = feature_cols + ["Weighted_MA_10", "Weighted_MA_20", "Weighted_MA_30", "Weighted_MA_40",
                                           "Weighted_MA_50"]
        if detectorsoption[4] == "1":
            feature_cols = feature_cols + ["MA_of_diff_10", "MA_of_diff_20", "MA_of_diff_30", "MA_of_diff_40",
                                           "MA_of_diff_50"]
        if detectorsoption[5]=="1":
            feature_cols = feature_cols +["EWMA_0.1", "EWMA_0.3", "EWMA_0.5", "EWMA_0.7","EWMA_0.9"]
        if detectorsoption[6] == "1":
            feature_cols = feature_cols + ["TSD_1", "TSD_2", "TSD_3", "TSD_4", "TSD_5"]
        if detectorsoption[7] == "1":
            feature_cols = feature_cols + ["TSD_MAD_1", "TSD_MAD_2", "TSD_MAD_3", "TSD_MAD_4", "TSD_MAD_5"]
        if detectorsoption[8]=="1":
            feature_cols = feature_cols +["Historical_average_1", "Historical_average_2", "Historical_average_3","Historical_average_4","Historical_average_5"]
        if detectorsoption[11] == "1":
            for i in range(1, 6):
                for j in range(1, 2):
                    feature_cols = feature_cols + ["SVD_"+str(i*10)+"rows_"+str(2*j+1)+"weeks"]
        features = df[feature_cols]
        target = df['Label'].astype('int')
        #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=0)

        #print(X_train.shape)

        clf = RandomForestClassifier(n_jobs=2)
        #scores = cross_val_score(clf, features, target, cv=5)
        kf = KFold(n_splits=5, shuffle=False)
        scores_ac = cross_val_score(clf, features, target,scoring='accuracy', cv=kf)
        scores_re = cross_val_score(clf, features, target, scoring='recall', cv=kf)
        scores_pr = cross_val_score(clf, features, target, scoring='precision', cv=kf)
        scores_f1 = cross_val_score(clf, features, target, scoring='f1', cv=kf)
        scores_predict = cross_val_predict(clf, features, target,  cv=kf)
        result = [scores_ac.mean(),scores_re.mean(),scores_pr.mean(),scores_f1.mean()]
        print(result)

        df.to_csv(inputfile.split(".")[0]+"feature_result.csv", sep=',', header=True, index=True)
        #file1 = open(inputfile.split(".")[0] + 'score_predict_result.txt', 'w')
        #file1.write(str(scores_predict))
        #file1.close()
        np.savetxt(inputfile.split(".")[0] + 'score_predict_result.txt', scores_predict, delimiter=',')
        file = open(inputfile.split(".")[0]+'mearsure_result.txt', 'w')
        file.write(str(result))
        file.close()
        return result
    except Exception as e:
        print(e)


def RFValueCrossValidation(df):
    #带交叉验证的随机森林
    feature_cols = ['Value']
    #feature_cols=copy.deepcopy(feature_head).remove('Timestamp')
    #feature_cols = ["Simple_MA_10","Simple_MA_20","Simple_MA_30","Simple_MA_40","Simple_MA_50"]
    features = df[feature_cols]
    target = df['Label']
    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=0)

    #print(X_train.shape)

    clf = RandomForestClassifier(n_jobs=2)
    scores = cross_val_score(clf, features, target, cv=5)
    result=scores.mean()
    print(result)
    return result


#=============================下面为具体的detector===================================================

def alg1(timestamp,df):
    pass



def calculateDiff(df, timestamp, parameter):#测试通过
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
        raise Exception('Timestamp out of index')
    else:
        return diff_value


def simple_ma(df, timestamp, win=10):#测试通过
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    try:
        part_df = df.loc[st:ed, "Value"]
        if len(part_df) != win:
            raise Exception
        return part_df.sum() / win
    except Exception as e :
        raise e





def weighted_ma(df, timestamp, win=10):#测试通过
    w = []
    cnt = 1
    for i in range(0, win):
        if i <= int(win / 4):
            w.append(cnt)
            cnt += 1
        elif i >= int(win * 3 / 4):
            cnt = cnt - 1
            w.append(cnt)
        else:
            w.append(cnt)
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    t = st
    val = 0
    i = 0
    try:
        while t <= ed:
            val += w[i] * df.loc[t, "Value"]
            i += 1
            t += 60
    except Exception:
        raise Exception
    return val / sum(w)


def ma_diff(df, timestamp, win=10):#测试通过
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    t = st
    diff = []
    try:
        while t <= ed:
            diff.append(df.loc[t, "Value"] - df.loc[t - 60 * 10, "Value"])
            t += 60
        return sta.mean(diff)
    except Exception as e:
        #print(e)
        #print(1)
        raise Exception


def calculateEWMA(df, timestamp, alpha):#测试通过
    new_df = df.loc[:timestamp, ['Value']]
    return new_df.ewm(alpha=alpha, adjust=False).mean().loc[timestamp, 'Value']


def calculateTSD(df, timestamp, win):#测试通过
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





def calculateTSDmad(df, timestamp, win):#测试通过
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


#根据已知miu和std的正态分布求x对应的概率密度
def norm(miu, std, x):
    u = (x - miu) / std
    x = abs(u) / math.sqrt(2)
    T = (0.0705230784, 0.0422820123, 0.0092705272,
         0.0001520143, 0.0002765672, 0.0000430638)
    E = 1 - pow((1 + sum([a * pow(x, (i + 1))
                          for i, a in enumerate(T)])), -16)
    p = 0.5 - 0.5 * E if u < 0 else 0.5 + 0.5 * E
    return p


def historical_average(df, timestamp, win=1):#测试通过
    number_of_points = win * 24 * 60
    st = timestamp - number_of_points * 60 + 60
    ed = timestamp
    try:
        val_list = df.loc[st:ed, "Value"].tolist()
        miu = sta.mean(val_list)
        if len(val_list) == 1:
            std = 0
        else:
            std = sta.stdev(val_list)
        val = df.loc[timestamp, "Value"]
        return 1 - norm(miu, std, val)
    except Exception:
        raise Exception


def alg10(timestamp,df):
    pass


def alg11(timestamp,df):
    pass


def svd(df, timestamp, row=10, col=3):#数据样本太少，太多时间戳的值为异常，测试未通过
    #当前点和前面10个点，前面3周
    try:
        a_list = []
        for num_of_day in range(col):  #col * 7
            st = timestamp - row * 60 - num_of_day * 24 * 60 #一天前（以天为周期）
            ed = timestamp - num_of_day * 24 * 60
            a_list.append(df.loc[st:ed, "Value"].tolist())
        #print(np.asarray(a_list).shape)
        u, s, vh = np.linalg.svd(np.asarray(a_list), full_matrices=False)
        matrix_s = np.diag(s)
        r = round(min(row, col) * 0.2) #col * 7
        for i in range(min(r, len(s))):
            matrix_s[i][i] = 0
        #print(matrix_s)
        residual_matrix = np.dot(np.dot(u, matrix_s), vh)
        last_column_list = [residual_matrix[i][col - 1] for i in range(1, col)]
        miu = np.mean(last_column_list)
        std = np.std(last_column_list)
        #print(miu, std, residual_matrix[0][col - 1])
        #print(norm(miu, std, residual_matrix[0][col - 1]))
        return norm(miu, std, residual_matrix[0][col - 1])
    except Exception as e:
        print(e)
    #    raise Exception


def alg13(timestamp,df):
    pass


def alg14(timestamp,df):
    pass


#=================================下面为detector当中用到的功能函数===================================

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

def test(input_file):
    [a,b,c,d] = dataload("dataSource\\Train\\Train\\train105.csv")
    [e,f,g] = featuregenerate(a,b,c)
    return RFCrossValidation(e, f, input_file)

if __name__ == '__main__':
    print([test("dataSource\\Train\\Train\\train101-m.csv"),
           test("dataSource\\Train\\Train\\train102.csv"),
           test("dataSource\\Train\\Train\\train103.csv"),
           test("dataSource\\Train\\Train\\train104.csv"),
    #       test("dataSource\\Train\\Train\\train105.csv"),
    #       test("dataSource\\Train\\Train\\train106.csv"),
    #       test("dataSource\\Train\\Train\\train107.csv"),
    #       test("dataSource\\Train\\Train\\train108.csv"),
    #       test("dataSource\\Train\\Train\\train109.csv"),
    #       test("dataSource\\Train\\Train\\train110.csv"),
    #       test("dataSource\\Train\\Train\\train111.csv"),
    #       test("dataSource\\Train\\Train\\train112.csv"),
    #       test("dataSource\\Train\\Train\\train113.csv"),
    #       test("dataSource\\Train\\Train\\train114.csv"),
    #       test("dataSource\\Train\\Train\\train115.csv")
           ])

    #[a,b,c,d] = dataload("dataSource\\Train\\Train\\train105.csv")
    #[e,f,g] = featuregenerate(a,b,c)
    #print(RFCrossValidation(e,f,"dataSource\\Train\\Train\\train105.csv"))


    #print(simple_ma(dataload()[0],1497427740,10))
    #print(featuregenerate(dataload()[0],dataload()[1],dataload()[2]))
    #print(dataload()[1].index)
    #print(RFValueCrossValidation(dataload()[3]))
    #print(RFtraining(dataload()[1]))
    #print(featuregenerate(dataload()[0],dataload()[1])[1])
    pass