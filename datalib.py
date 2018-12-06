#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:37:56 2018

@author: igor
"""

import talib.abstract as ta
import requests
import secrets as s
import datetime as dt
import params as p
import os
import pickle
import pandas as pd


# Load Historical Price Data from Cryptocompare
# API Guide: https://medium.com/@agalea91/cryptocompare-api-quick-start-guide-ca4430a484d4
def load_data():
    now = dt.datetime.today().strftime('%Y-%m-%d')
    if (not p.reload) and os.path.isfile(p.file): 
        df = pickle.load(open(p.file, "rb" ))
        # Return loaded price data if it is up to date
        if df.date.iloc[-1].strftime('%Y-%m-%d') == now:
            print('Using loaded prices for ' + now)
            return df
    
    if p.bar_period == 'day':
        period = 'histoday'
    elif p.bar_period == 'hour': 
        period = 'histohour'
    
    retry = True
    while retry: # This is to avoid issue when only 31 rows are returned
        r = requests.get('https://min-api.cryptocompare.com/data/'+period
                         +'?fsym='+p.ticker+'&tsym='+p.currency
                         +'&allData=true&e='+p.exchange
                         +'&api_key='+s.cryptocompare_key)
        df = pd.DataFrame(r.json()['Data'])
        if len(df) > p.min_data_size: 
            retry = False
        else:
            print("Incomplete price data. Retrying ...")
    df = df.set_index('time')
    df['date'] = pd.to_datetime(df.index, unit='s')
    os.makedirs(os.path.dirname(p.file), exist_ok=True)
    pickle.dump(df, open(p.file, "wb" ))
    print('Loaded Prices. Period:'+p.bar_period+' Rows:'+str(len(df))+' Date:'+str(df.date.iloc[-1]))
    return df

def load_prices():
    """ Loads hourly historical prices and converts them to daily usung p.time_offset
        Stores hourly prices in price.csv
        Returns DataFrame with daily price data
    """
    has_data = True
    min_time = 0
    first_call = True
    file = p.cfgdir+'/price.csv'
    if p.reload or not os.path.isfile(file):
        while has_data:
            url = ('https://min-api.cryptocompare.com/data/histohour'
                +'?fsym='+p.ticker+'&tsym='+p.currency
                +'&e='+p.exchange
                +'&limit=10000'
                +'&api_key='+s.cryptocompare_key
                +('' if first_call else '&toTs='+str(min_time)))
                             
            r = requests.get(url)
            df = pd.DataFrame(r.json()['Data'])
            if df.close.max() == 0 or len(df) == 0:
                has_data = False
            else:
                min_time = df.time[0] - 1
                with open(file, 'w' if first_call else 'a') as f: 
                    df.to_csv(f, header=first_call, index = False)
            
            if first_call: first_call = False
        print('Loaded Hourly Prices in UTC')

    df = pd.read_csv(file)
    df = df.set_index('time')
    df = df[df.close > 0]  
    df['date'] = pd.to_datetime(df.index, unit='s')
    if p.time_lag > 0:
        df['date'] = df.date - dt.timedelta(hours=p.time_lag)
        df = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 
            'volumefrom': 'sum', 'volumeto': 'sum'})
    print('Price Rows: '+str(len(df))+' Last Timestamp: '+str(df.date.max()))
    return df

# Map feature values to bins (numbers)
# Each bin has same number of feature values
def bin_feature(feature, bins=None, cum=True):
    if bins is None: bins = p.feature_bins
    l = lambda x: int(x[x < x[-1]].size/(x.size/bins))
    if cum:
        return feature.expanding().apply(l, raw = True)
    else:
        return ((feature.rank()-1)/(feature.size/bins)).astype('int')

#    binfile = p.cfgdir+'/bin'+feature.name+'.pkl'
#    if test:
#        b = pickle.load(open(binfile, "rb" )) # Load bin config
#        d = pd.cut(feature, bins=b, labels=False, include_lowest=True)
#    else:
#        d, b = pd.qcut(feature, bins, duplicates='drop', labels=False, retbins=True)
##        d, b = pd.qcut(feature.rank(method='first'), bins, labels=False, retbins=True)
#        pickle.dump(b, open(binfile, "wb" )) # Save bin config
#    return d

# Read Price Data and add features
def get_dataset(test=False):
    df = pickle.load(open(p.file, "rb" ))
    
    # Add features to dataframe
    # Typical Features: close/sma, bollinger band, holding stock, return since entry
    df['dr'] = df.close/df.close.shift(1)-1 # daily return
    df['adr'] = ta.SMA(df, price='dr', timeperiod=p.adr_period)
    df['sma'] = ta.SMA(df, price='close', timeperiod=p.sma_period)
    df['dsma'] = df.sma/df.sma.shift(1)-1
    df['rsma'] = df.close/df.sma
    df['rsi'] = ta.RSI(df, price='close', timeperiod=p.rsi_period)
    df['hh'] = df.high/ta.MAX(df, price='high', timeperiod=p.hh_period)
    df['ll'] = df.low/ta.MIN(df, price='low', timeperiod=p.ll_period)
    df['hhll'] = (df.high+df.low)/(df.high/df.hh+df.low/df.ll)
    df = df.dropna()
    # Map features to bins
    df = df.assign(binrsi=bin_feature(df.rsi))
    if p.version == 1:
        df = df.assign(binadr=bin_feature(df.adr))
        df = df.assign(binhh=bin_feature(df.hh))
        df = df.assign(binll=bin_feature(df.ll))
    elif p.version == 2:
        df = df.assign(bindsma=bin_feature(df.dsma))
        df = df.assign(binrsma=bin_feature(df.rsma))
        df = df.assign(binhhll=bin_feature(df.hhll))
    
    if p.max_bars > 0: df = df.tail(p.max_bars).reset_index(drop=True)
    # Separate Train / Test Datasets using train_pct number of rows
    if test:
        rows = int(len(df)*p.test_pct)
        return df.tail(rows).reset_index(drop=True)
    else:
        rows = int(len(df)*p.train_pct)
        return df.head(rows).reset_index(drop=True)

# Sharpe Ratio Calculation
# See also: https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement
def get_sr(df):
    return df.mean()/(df.std()+0.000000000000001) # Add small number to avoid division by 0

def get_ret(df):
    return df.iloc[-1]/df.iloc[0]

def normalize(df):
    return df/df.iloc[0]

