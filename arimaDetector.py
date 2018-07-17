import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import datetime
import pickle

def calculateARIMA(df, timestamp, modelFilename):  #  从文件中读取已经训练好的模型用于predict
    with open(modelFilename, "rb") as pkf:
        model = pickle.load(pkf)
    timestamp = pd.to_datetime(timestamp, unit='s')
    predict_data = float(model.predict(start=timestamp, end=timestamp, dynamic=False))
    return predict_data