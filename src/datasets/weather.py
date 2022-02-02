# -*- coding: utf-8 -*-
import json
import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def dayofweek_encoder(row):
    dr = pd.date_range(start='2019-01-01', end='2019-12-31')
    cal = calendar()
    holidays = cal.holidays(start=dr.min(), end=dr.max())

    if row['date'] in holidays or row['dow'] in [5,6]:
        row['dow_wkd_holiday'] = 1
    elif row['dow'] == 0:
        row['dow_mon'] = 1
    elif row['dow'] in [1,2,3]:
        row['dow_tue_thr'] = 1
    else:
        row['dow_fri'] = 1
    return row

def read_data(path):
        with open(path, "r") as read_file:
            data = json.load(read_file)
        df = pd.DataFrame(data["observations"])
        return df
    
def weatherData():
    List = []
    for month in ['01','02','03','04','05','06','07','08','09','10','11','12']: 
        path = 'data/{}01.json'.format(month)
        
        df = read_data(path)
        List.append(df)
    
    df = pd.concat(List)
    df['valid_time_gmt'] = df.copy()['valid_time_gmt'].map(lambda x: datetime.datetime.fromtimestamp(int(x)))
    df.index = df.valid_time_gmt
    mask = df.index.duplicated()
    df = df.loc[~mask, ['vis','wspd','rh','pressure','precip_hrly','temp']].resample('5min').ffill()
    df = df.loc['2019-02-10 00:00:00':'2019-07-23 23:55:00']
        
    zscore = preprocessing.StandardScaler()
    df_zscore = zscore.fit_transform(df)
    df = pd.DataFrame(df_zscore,index=df.index,columns=df.columns)
    return df

