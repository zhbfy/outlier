import configparser
import pandas as pd
import numpy as np
import copy
import random
import traceback
from sklearn.model_selection  import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier


def dataload():
    #加载数据
    df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train101-m.csv",sep=',')
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
    conf.read("E:\\code-exercise\\outlierDetection\\detectors\\DetectorsConfig")
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


    feature_head =["Timestamp"]#特征表头
    if detectorsoption[2]=="1":
        feature_head.append("Simple_MA_10")
        feature_head.append("Simple_MA_20")
        feature_head.append("Simple_MA_30")
        feature_head.append("Simple_MA_40")
        feature_head.append("Simple_MA_50")
    feature = pd.DataFrame(columns=feature_head) #生成特征表

    for index,row in df_positive.iterrows():#生成正例特征
        try:

            feature_insert = []
            index=int(index)
            feature_insert.append(index)
            if detectorsoption[2]=="1":
                f1 = [simple_ma(df_new,index,10)]
                f2 = [simple_ma(df_new,index, 20)]
                f3 = [simple_ma(df_new,index, 30)]
                f4 = [simple_ma(df_new,index, 40)]
                f5 = [simple_ma(df_new,index, 50)]
                feature_insert.append(f1)
                feature_insert.append(f2)
                feature_insert.append(f3)
                feature_insert.append(f4)
                feature_insert.append(f5)
        except Exception as e:
            print('Exception: ', e)
        else:
            insertRow = pd.DataFrame([feature_insert], columns=feature_head)
            feature.append(insertRow,ignore_index=True)

    trainingdata_num = copy.deepcopy(feature.iloc[:, 0].size)*2 #训练样本条数
    while feature.iloc[:,0].size<trainingdata_num:
        try:
            index= df_wholenegative.irow[random.randint(0,df_wholenegative.iloc[:,0].size-1)].index
            feature_insert = []
            feature_insert.append(index)
            if detectorsoption[2]=="1":
                f1 = [simple_ma(df_new,index,10)]
                f2 = [simple_ma(df_new, index, 20)]
                f3 = [simple_ma(df_new, index, 30)]
                f4 = [simple_ma(df_new, index, 40)]
                f5 = [simple_ma(df_new, index, 50)]
                feature_insert.append(f1)
                feature_insert.append(f2)
                feature_insert.append(f3)
                feature_insert.append(f4)
                feature_insert.append(f5)
        except:
            pass
        else:
            insertRow = pd.DataFrame([feature_insert], columns=feature_head)
            feature.append(insertRow,ignore_index=True)
    return [detectorsoption,feature]


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

def RFCrossValidation(df):
    #带交叉验证的随机森林
    feature_cols = ['Value']
    features = df[feature_cols]
    target = df.Label
    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=0)

    #print(X_train.shape)

    clf = RandomForestClassifier(n_jobs=2)
    scores = cross_val_score(clf, features, target, cv=5)
    return scores




def simple_ma(df, timestamp, win=10):
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    part_df = df.loc[st:ed, "Value"]
    if len(part_df) != win:
        raise Exception
    return part_df.sum() / win


def alg2(timestamp,df):
    pass


def alg3(timestamp,df):
    pass


def alg4(timestamp,df):
    pass


def alg5(timestamp,df):
    pass


def alg6(timestamp,df):
    pass


def alg7(timestamp,df):
    pass


def alg8(timestamp,df):
    pass


def alg9(timestamp,df):
    pass


def alg10(timestamp,df):
    pass


def alg11(timestamp,df):
    pass


def alg12(timestamp,df):
    pass


def alg13(timestamp,df):
    pass


def alg14(timestamp,df):
    pass


if __name__ == '__main__':
    print(simple_ma(dataload()[0],1497427740,10))
    #print(featuregenerate(dataload()[0],dataload()[1],dataload()[2]))
    #print(dataload()[1].index)
    #print(RFCrossValidation(dataload()[3]))
    #print(RFtraining(dataload()[1]))
    #print(featuregenerate(dataload()[0],dataload()[1])[1])
    pass