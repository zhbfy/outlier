import configparser
import pandas as pd
import numpy as np
import copy
import statistics as sta
import math
import random
import traceback

def unfoundedOutlier(inputfeaturefile,predictfile,inputsourcefile):
    df = pd.read_csv(inputfeaturefile, sep=',')

    df1 = pd.read_csv(predictfile, header=None,names=["label"])
    df2 = pd.read_csv(inputsourcefile, sep=',')

    begin =[]
    end =[]

    df["predictedlabel"] = df1["label"]
    df2["outliernumber"]= None
    j=0 #第几段异常
    k=0 #是否异常标志位
    df["outlier_numbers"]=np.nan

    #print(df)
    if df2.iloc[0,2]==0:
        k=0
    else:
        k=1
        j=1
        re = int(df2.loc[0]["Timestamp"])
        begin.append(re)
        #print(df2.loc[0]["Timestamp"])
    #df2.iloc[0,3]=j
    for i in range(1, len(df2)):
        try:
            if k==0:
                if (df2.iloc[i,2]-df2.iloc[i-1,2])==1:
                    k=k+1
                    j=j+1
                    re=int(df2.loc[i]["Timestamp"])
                    begin.append(re)
                    #print(df2.loc[i]["Timestamp"])
                #df2.iloc[i,3] = j
                if (df2.iloc[i,2]-df2.iloc[i-1,2])==0:

                    pass
                #df2.iloc[i,3] = k
            if k==1:
                if (df2.iloc[i,2]-df2.iloc[i-1,2])==-1:
                    k=0
                #df2.iloc[i,3] = k
                #print(i)
                    ed = int(df2.loc[i-1]["Timestamp"])
                    end.append(ed)
                    #print(df2.iloc[i-1]["Timestamp"])
                if (df2.iloc[i,2]-df2.iloc[i-1,2])==0:
                    #df2.iloc[i,3] = j
                    if i == len(df2) - 1:
                        ed = int(df2.loc[i - 1]["Timestamp"])
                        end.append(ed)
                        #print(df2.iloc[i - 1]["Timestamp"])
        except Exception as e:
            print(e)
    flag=0
    result=0

    for j in  range (0,len(begin)):
        for l in range (0, len(df)):
          if df.iloc[l,0]>=begin[j] and  df.iloc[l,0]<=end[j] and df.iloc[l,1]==df.iloc[l,40]:
              flag=1
        if flag==0:
            result=result+1
    print(len(begin),result)
    #print(df2)

    #a = pd.DataFrame(df["Label"])
    #a = a.append([0], ignore_index=True)
    #b= pd.DataFrame(df1["label"])
    #b = b.append([0], ignore_index=True)

    #for i in range(0, len(df)):













    #result = pd.DataFrame(columns=["Label","predict","Predict-Real","unfonudedOutlier"])
    #for indexs in df.index:
    #    print(df.loc[indexs].values[0:-1])


if __name__ == '__main__':
    unfoundedOutlier("test-mid-result-good-2\\data-101feature_result.csv",
                     "test-mid-result-good-2\\data-101score_predict_result.txt",
                     "dataSource\\Train\\Train\\train101-m.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-102feature_result.csv",
                     "test-mid-result-good-2\\data-102score_predict_result.txt",
                     "dataSource\\Train\\Train\\train102.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-103feature_result.csv",
                     "test-mid-result-good-2\\data-103score_predict_result.txt",
                     "dataSource\\Train\\Train\\train103.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-104feature_result.csv",
                     "test-mid-result-good-2\\data-104score_predict_result.txt",
                     "dataSource\\Train\\Train\\train104.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-105feature_result.csv",
                     "test-mid-result-good-2\\data-105score_predict_result.txt",
                     "dataSource\\Train\\Train\\train105.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-106feature_result.csv",
                     "test-mid-result-good-2\\data-106score_predict_result.txt",
                     "dataSource\\Train\\Train\\train106.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-107feature_result.csv",
                     "test-mid-result-good-2\\data-107score_predict_result.txt",
                     "dataSource\\Train\\Train\\train107.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-108feature_result.csv",
                     "test-mid-result-good-2\\data-108score_predict_result.txt",
                     "dataSource\\Train\\Train\\train108.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-109feature_result.csv",
                     "test-mid-result-good-2\\data-109score_predict_result.txt",
                     "dataSource\\Train\\Train\\train109.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-110feature_result.csv",
                     "test-mid-result-good-2\\data-110score_predict_result.txt",
                     "dataSource\\Train\\Train\\train110.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-111feature_result.csv",
                     "test-mid-result-good-2\\data-111score_predict_result.txt",
                     "dataSource\\Train\\Train\\train111.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-112feature_result.csv",
                     "test-mid-result-good-2\\data-112score_predict_result.txt",
                     "dataSource\\Train\\Train\\train112.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-113feature_result.csv",
                     "test-mid-result-good-2\\data-113score_predict_result.txt",
                     "dataSource\\Train\\Train\\train113.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-114feature_result.csv",
                     "test-mid-result-good-2\\data-114score_predict_result.txt",
                     "dataSource\\Train\\Train\\train114.csv")
    unfoundedOutlier("test-mid-result-good-2\\data-115feature_result.csv",
                     "test-mid-result-good-2\\data-115score_predict_result.txt",
                     "dataSource\\Train\\Train\\train115.csv")
