import pandas as pd
import statistics as sta
import math
import numpy as np

def dataload():
    #加载数据
    #df = pd.read_csv(inputfile,sep=',')
    df = pd.read_csv("E:\\code-exercise\\outlierDetection\\dataSource\\Train\\Train\\train101-m.csv",sep=',')
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

def simple_ma(df, timestamp, win=10):
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    try:
        part_df = df.loc[st:ed, "Value"]
        if len(part_df) != win:
            raise Exception
        return part_df.sum() / win
    except Exception:
        raise Exception


#df = dataload()[0]
#print(simple_ma(df, 1497116640, 10))


def weighted_ma(df, timestamp, win=10):
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


#df = dataload()[0]
#print(weighted_ma(df, 1497116640, 10))


def ma_diff(df, timestamp, win=10):
    st = timestamp - int(win / 2) * 60
    ed = timestamp + (win - int(win / 2) - 1) * 60
    t = st
    diff = []
    try:
        while t <= ed:
            if len(df[df[0] == t][1]) != 1:
                raise Exception
            if len(df[df[0] == t - 60]) != 1:
                raise Exception
            diff.append(df.loc[t, "Value"] - df.loc[t - 60 * 10, "Value"])
            t += 60
        return sta.mean(diff)
    except Exception:
        raise Exception


#df = dataload()[0]
#print(ma_diff(df, 1400000060))

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


def historical_average(df, timestamp, win=1):
    number_of_points = win * 7 * 24 * 60
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

#df = dataload()[0]
#print(historical_average(df, 1502685720,5))


def svd(df, timestamp, row=10, col=3):
    #当前点和前面10个点，前面3周
    try:
        a_list = []
        for num_of_day in range(col):  #col * 7
            st = timestamp - row * 60 - num_of_day * 24 * 60 #一天前（以天为周期）
            ed = timestamp - num_of_day * 24 * 60
            a_list.append(df.loc[st:ed, "Value"].tolist())
        print(np.asarray(a_list).shape)
        u, s, vh = np.linalg.svd(np.asarray(a_list), full_matrices=False)
        matrix_s = np.diag(s)
        r = round(min(row, col) * 0.2) #col * 7
        for i in range(min(r, len(s))):
            matrix_s[i][i] = 0
        print(matrix_s)
        residual_matrix = np.dot(np.dot(u, matrix_s), vh)
        last_column_list = [residual_matrix[i][col - 1] for i in range(1, col)]
        miu = np.mean(last_column_list)
        std = np.std(last_column_list)
        print(miu, std, residual_matrix[0][col - 1])
        print(norm(miu, std, residual_matrix[0][col - 1]))
        return norm(miu, std, residual_matrix[0][col - 1])
    except Exception:
        raise Exception


#df = dataload()[0]
#print(svd(df, 1502685720))

def holtwinters(df, timestamp, alpha, beta, gamma):
        c = 24 * 60
    #try:
        y = df.loc[:timestamp, "Value"].tolist()

        # Compute initial b and intercept using the first two complete c periods.
        ylen = len(y)
        if ylen % c != 0:
            raise Exception
        fc = float(c)
        ybar2 = sum([y[i] for i in range(c, 2 * c)]) / fc
        ybar1 = sum([y[i] for i in range(c)]) / fc
        b0 = (ybar2 - ybar1) / fc

        # Compute for the level estimate a0 using b0 above.
        tbar = sum(i for i in range(1, c + 1)) / fc
        a0 = ybar1 - b0 * tbar

        # Compute for initial indices
        I = [y[i] / (a0 + (i + 1) * b0) for i in range(0, ylen)]

        S = [0] * (ylen + c)
        for i in range(c):
            S[i] = (I[i] + I[i + c]) / 2.0

        # Normalize so S[i] for i in [0, c)  will add to c.
        tS = c / sum([S[i] for i in range(c)])
        for i in range(c):
            S[i] *= tS

        F = [0] * (ylen + c)

        At = a0
        Bt = b0
        for i in range(ylen):
            Atm1 = At
            Btm1 = Bt
            At = alpha * y[i] / S[i] + (1.0 - alpha) * (Atm1 + Btm1)
            Bt = beta * (At - Atm1) + (1 - beta) * Btm1
            S[i + c] = gamma * y[i] / At + (1.0 - gamma) * S[i]
            F[i] = (a0 + b0 * (i + 1)) * S[i]
            #print("i=", i + 1, "y=", y[i], "S=", S[i], "Atm1=", Atm1, "Btm1=", Btm1, "At=", At, "Bt=", Bt, "S[i+c]=", S[i + c], "F=", F[i])
            #print(i, y[i], F[i])
        # Forecast for next c periods:
        #for m in range(c):
        #    print("forecast:", (At + Bt * (m + 1)) * S[ylen + m])
        forecast_val = (At + Bt) * S[ylen]
        return abs(forecast_val - df.loc[timestamp, "Value"])
    #except Exception:
    #    raise Exception

df = dataload()[0]
print(holtwinters(df, 1497116640,0.2,0.2,0.2))