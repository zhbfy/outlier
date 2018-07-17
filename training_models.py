import configparser
import pandas as pd
import numpy as np
import copy
import statistics as sta
import math
import random
import traceback

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import matplotlib.pylab as plt
import datetime
from sklearn.model_selection  import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier

def parser():
    #解析文件
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
    return detectorsoption

def RFCrossValidation(detectorsoption,df,inputfile):
    #带交叉验证的随机森林
    #feature_cols = ['Value']
    #feature_cols=copy.deepcopy(feature_head).remove('Timestamp')
    df = df.set_index("Timestamp").sort_index(ascending=True)
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

        clf = RandomForestClassifier(n_estimators=50,n_jobs=2,oob_score=True)
        #scores = cross_val_score(clf, features, target, cv=5)
        #kf = KFold(n_splits=5, shuffle=False)
        kf = StratifiedKFold(n_splits=5, shuffle=False)
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

def training(inputfeaturefile,datasetname):
    df = pd.read_csv(inputfeaturefile, sep=',')

    RFCrossValidation(parser(),df,"test-mid-result-good-2\\"+datasetname)

if __name__ == '__main__':
    training("test-mid-result\\train101-mfeature_result.csv","data-101.csv")
    training("test-mid-result\\train102feature_result.csv", "data-102.csv")
    training("test-mid-result\\train103feature_result.csv", "data-103.csv")
    training("test-mid-result\\train104feature_result.csv", "data-104.csv")
    training("test-mid-result\\train105feature_result.csv", "data-105.csv")
    training("test-mid-result\\train106feature_result.csv", "data-106.csv")
    training("test-mid-result\\train107feature_result.csv", "data-107.csv")
    training("test-mid-result\\train108feature_result.csv", "data-108.csv")
    training("test-mid-result\\train109feature_result.csv", "data-109.csv")
    training("test-mid-result\\train110feature_result.csv", "data-110.csv")
    training("test-mid-result\\train111feature_result.csv", "data-111.csv")
    training("test-mid-result\\train112feature_result.csv", "data-112.csv")
    training("test-mid-result\\train113feature_result.csv", "data-113.csv")
    training("test-mid-result\\train114feature_result.csv", "data-114.csv")
    training("test-mid-result\\train115feature_result.csv", "data-115.csv")