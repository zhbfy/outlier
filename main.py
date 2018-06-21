import configparser
import pandas as pd
import numpy as np
import copy
import random
import traceback
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


    feature_head =["Timestamp","Label"]#特征表头
    if detectorsoption[1]=="1":
        feature_head = feature_head + ["Diff_1_slot", "Diff_1_day", "Diff_1_week"]
    if detectorsoption[2]=="1":
        feature_head = feature_head+["Simple_MA_10","Simple_MA_20","Simple_MA_30","Simple_MA_40","Simple_MA_50"]
    if detectorsoption[5] == "1":
        feature_head = feature_head + ["EWMA_0.1", "EWMA_0.3", "EWMA_0.5", "EWMA_0.7","EWMA_0.9"]
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

            if detectorsoption[5] == "1":  # EWMA
                f51 = calculateEWMA(df_new, index, 0.1)
                f52 = calculateEWMA(df_new, index, 0.3)
                f53 = calculateEWMA(df_new, index, 0.5)
                f54 = calculateEWMA(df_new, index, 0.7)
                f55 = calculateEWMA(df_new, index, 0.9)
                feature_insert = feature_insert + [f51, f52, f53, f54, f55]

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
            if detectorsoption[5] == "1":  # EWMA
                f51 = calculateEWMA(df_new, index, 0.1)
                f52 = calculateEWMA(df_new, index, 0.3)
                f53 = calculateEWMA(df_new, index, 0.5)
                f54 = calculateEWMA(df_new, index, 0.7)
                f55 = calculateEWMA(df_new, index, 0.9)
                feature_insert = feature_insert + [f51, f52, f53, f54, f55]
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

def RFCrossValidation(detectorsoption,df,feature_head):
    #带交叉验证的随机森林
    #feature_cols = ['Value']
    #feature_cols=copy.deepcopy(feature_head).remove('Timestamp')
    try:
        feature_cols=[]
        if detectorsoption[1]=="1":
            feature_cols = feature_cols +["Diff_1_slot", "Diff_1_day", "Diff_1_week"]
        if detectorsoption[2]=="1":
            feature_cols = feature_cols +["Simple_MA_10","Simple_MA_20","Simple_MA_30","Simple_MA_40","Simple_MA_50"]
        if detectorsoption[5]=="1":
            feature_cols = feature_cols +["EWMA_0.1", "EWMA_0.3", "EWMA_0.5", "EWMA_0.7","EWMA_0.9"]
        features = df[feature_cols]
        target = df['Label'].astype('int')
        #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=0)

        #print(X_train.shape)

        clf = RandomForestClassifier(n_jobs=2)
        scores = cross_val_score(clf, features, target, cv=5)
        return scores.mean()
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
    return scores.mean()




def alg1(timestamp,df):
    pass



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
        raise Exception('Timestamp out of index')
    else:
        return diff_value


def simple_ma(df, timestamp, win=10):
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    try:
        part_df = df.loc[st:ed, "Value"]
        if len(part_df) != win:
            raise Exception
        return part_df.sum() / win
    except Exception as e :
        raise e





def alg4(timestamp,df):
    pass


def alg5(timestamp,df):
    pass


def calculateEWMA(df, timestamp, alpha):
    new_df = df.loc[:timestamp, ['Value']]
    return new_df.ewm(alpha=alpha, adjust=False).mean().loc[timestamp, 'Value']


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

def test(input_file):
    [a, b, c, d] = dataload(input_file)
    [e, f, g] = featuregenerate(a, b, c)
    return RFCrossValidation(e, f, g)

if __name__ == '__main__':
    print([test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train101-m.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train102.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train103.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train104.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train105.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train106.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train107.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train108.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train109.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train110.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train111.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train112.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train113.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train114.csv"),
           test("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train115.csv"),])

    #[a,b,c,d] = dataload()
    #[e,f,g] = featuregenerate(a,b,c)
    #print(RFCrossValidation(e,f,g))


    #print(simple_ma(dataload()[0],1497427740,10))
    #print(featuregenerate(dataload()[0],dataload()[1],dataload()[2]))
    #print(dataload()[1].index)
    #print(RFValueCrossValidation(dataload()[3]))
    #print(RFtraining(dataload()[1]))
    #print(featuregenerate(dataload()[0],dataload()[1])[1])
    pass