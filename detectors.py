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



def calculateEWMA(df, timestamp, alpha):
    new_df = df.loc[:timestamp, ['Value']]
    return new_df.ewm(alpha=alpha, adjust=False).mean().loc[timestamp, 'Value']