#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:28:17 2018

@author: igor
"""

import datetime as dt
import params as p
import talib
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout
from sklearn.preprocessing import StandardScaler
#from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import stats as s
import datalib as dl
#from sklearn.externals.joblib import dump, load
from joblib import dump, load
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_signal_str(s=''):
    if s == '': s = get_signal()
    txt = p.pair + ':'
    txt += ' NEW TRADE' if s['new_trade'] else '' 
    txt += ' Signal: ' + s['action'] 
    if p.short and s['action'] == 'Sell': txt += ' SHORT'
    txt += ' Open: '+str(s['open'])
    txt += ' P/L: '+str(s['pnl'])+'%'
    if s['tp']: txt += ' TAKE PROFIT!'
    if s['sl']: txt += ' STOP LOSS!'
    
    return txt 

def get_signal(offset=-1):
    s = td.iloc[offset]
    pnl = round(100*(s.ctrf - 1), 2)
    sl = p.truncate(s.sl_price, p.price_precision)
    tp = p.truncate(s.tp_price, p.price_precision)
    
    return {'new_trade':s.new_trade, 'action':s.signal, 
            'open':s.open, 'open_ts':s.date, 
            'close':s.close, 'close_ts':s.date_to, 'pnl':pnl, 
            'sl':s.sl, 'sl_price':sl, 'tp':s.tp, 'tp_price':tp}

def add_features(ds):
#    print('*** Adding Features ***')
    ds['VOL'] = ds['volume']/ds['volume'].rolling(window = p.vol_period).mean()
    ds['HH'] = ds['high']/ds['high'].rolling(window = p.hh_period).max() 
    ds['LL'] = ds['low']/ds['low'].rolling(window = p.ll_period).min()
    ds['DR'] = ds['close']/ds['close'].shift(1)
    ds['MA'] = ds['close']/ds['close'].rolling(window = p.sma_period).mean()
    ds['MA2'] = ds['close']/ds['close'].rolling(window = 2*p.sma_period).mean()
    ds['STD']= ds['close'].rolling(p.std_period).std()/ds['close']
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['WR'] = talib.WILLR(ds['high'].values, ds['low'].values, ds['close'].values, p.wil_period)
    ds['DMA'] = ds.MA/ds.MA.shift(1)
    ds['MAR'] = ds.MA/ds.MA2
    ds['Price_Rise'] = np.where(ds['DR'] > 1, 1, 0)

    ds = ds.dropna()
    
    return ds

def get_train_test(X, y):
    # Separate train from test
    train_split = int(len(X)*p.train_pct)
    test_split = p.test_bars if p.test_bars > 0 else int(len(X)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]
    
    # Feature Scaling
    # Load scaler from file for test run
#    from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
    scaler = p.cfgdir+'/sc.dmp'
    if p.train:
#        sc = QuantileTransformer(10)
#        sc = MinMaxScaler()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        dump(sc, scaler)
    else:
        sc = load(scaler)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        
    return X_train, X_test, y_train, y_test

def plot_fit_history(h):
    # Plot model history
    # Accuracy: % of correct predictions 
#    plt.plot(h.history['acc'], label='Train Accuracy')
#    plt.plot(h.history['val_acc'], label='Test Accuracy')
    plt.plot(h.history['loss'], label='Train')
    plt.plot(h.history['val_loss'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(X_train, X_test, y_train, y_test, file):
    print('*** Training model with '+str(p.units)+' units per layer ***')
    nn = Sequential()
    nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu'))
    nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    cp = ModelCheckpoint(file, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    nn.compile(optimizer = 'adam', loss = p.loss, metrics = ['accuracy'])
    history = nn.fit(X_train, y_train, batch_size = 100, 
                             epochs = p.epochs, callbacks=[cp], 
                             validation_data=(X_test, y_test), 
                             verbose=0)

    # Plot model history
    plot_fit_history(history)

    # Load Best Model
    nn = load_model(file) 
    
    return nn
    
# TODO: Use Long / Short / Cash signals
def gen_signal(ds, y_pred_val):
    td = ds.copy()
    td = td[-len(y_pred_val):]
    td['y_pred_val'] = y_pred_val
    td['y_pred'] = (td['y_pred_val'] >= p.signal_threshold)
    td = td.dropna()

    td['y_pred_id'] = np.trunc(td['y_pred_val'] * 1000)
    td['signal'] = td['y_pred'].map({True: 'Buy', False: 'Sell'})
    if p.ignore_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.ignore_signals), np.NaN, td.signal)
        td['signal'] = td.signal.fillna(method='ffill')
    if p.hold_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.hold_signals), 'Hold', td.signal)

    return td

def run_pnl(td, file):
    bt = td[['date','open','high','low','close','volume','signal']].copy()

    # Calculate Pivot Points
    bt['PP'] = (bt.high + bt.low + bt.close)/3
    bt['R1'] = 2*bt.PP - bt.low 
    bt['S1'] = 2*bt.PP - bt.high
    bt['R2'] = bt.PP + bt.high - bt.low
    bt['S2'] = bt.PP - bt.high + bt.low
    bt['R3'] = bt.high + 2*(bt.PP - bt.low)
    bt['S3'] = bt.low - 2*(bt.high - bt.PP)
    bt['R4'] = bt.high + 3*(bt.PP - bt.low)
    bt['S4'] = bt.low - 3*(bt.high - bt.PP)

    # Calculate SL price
    bt['sl_price'] = np.where(bt.signal=='Buy', bt.close.rolling(50).mean().shift(1), 0)
    bt['sl_price'] = np.where(bt.signal=='Sell', bt.R2.shift(1), bt.sl_price)
    bt['sl'] = False
    if p.buy_sl:   
        bt['sl'] = np.where((bt.signal=='Buy')&(bt.sl_price<=bt.open)&(bt.sl_price>=bt.low), True, bt.sl)
    if p.sell_sl:
        bt['sl'] = np.where((bt.signal=='Sell')&(bt.sl_price>=bt.open)&(bt.sl_price<=bt.high), True, bt.sl)
    
    # Calculate TP price
    bt['tp_price'] = np.where(bt.signal == 'Buy', bt.open * (1 + p.take_profit), 0)
    bt['tp_price'] = np.where(bt.signal == 'Sell', bt.open * (1 - p.take_profit), bt.tp_price)
    bt['tp'] = False 
    if p.buy_tp:
        bt['tp'] = np.where((bt.signal=='Buy')&(bt.tp_price>=bt.open)&(bt.tp_price<=bt.high), True, bt.tp) 
    if p.sell_tp and p.short:
        bt['tp'] = np.where((bt.signal=='Sell')&(bt.tp_price<=bt.open)&(bt.tp_price>=bt.low), True, bt.tp)    
    bt['new_trade'] = (bt.signal != bt.signal.shift(1)) | bt.sl.shift(1) | bt.tp.shift(1)
    bt['trade_id'] = np.where(bt.new_trade, bt.index, np.NaN)
    bt['open_price'] = np.where(bt.new_trade, bt.open, np.NaN)
    bt = bt.fillna(method='ffill')

    # SL takes precedence over TP if both are happening in same timeframe
    bt['close_price'] = np.where(bt.tp, bt.tp_price, bt.close)
    bt['close_price'] = np.where(bt.sl, bt.sl_price, bt.close_price)   

    # Rolling Trade Return
    bt['ctr'] = np.where(bt.signal == 'Buy', bt.close_price/bt.open_price, 1)
    if p.short: bt['ctr'] = np.where(bt.signal == 'Sell', 2 - bt.close_price/bt.open_price, bt.ctr)
    # Breakout: Buy if SL is triggered for Sell trade
    if p.breakout: bt['ctr'] = np.where((bt.signal == 'Sell') & bt.sl, bt.ctr*(bt.close/bt.sl_price), bt.ctr)

    # Margin Calculation. Assuming marging is used for short trades only
    bt['margin'] = 0
    if p.short:
        bt['margin'] = np.where(bt['signal'] == 'Sell',  p.margin, bt.margin)
        bt['margin'] = np.where(bt.new_trade & (bt['signal'] == 'Sell'), p.margin + p.margin_open, bt.margin)
    
    bt['summargin'] = bt.groupby('trade_id')['margin'].transform(pd.Series.cumsum)

    # Rolling Trade Open and Close Fees
    bt['fee'] = p.limit_fee + bt.ctr*p.limit_fee
    bt['fee'] = np.where(bt.sl, p.limit_fee + bt.ctr*p.market_fee, bt.fee)
    if p.short:
        if p.breakout: 
            bt['fee'] = np.where((bt.signal == 'Sell') & bt.sl, bt.fee + bt.ctr*(p.market_fee + p.limit_fee), bt.fee)
    else:
        if not p.breakout: bt['fee'] = np.where(bt.signal == 'Sell', 0, bt.fee)
    
    # Rolling Trade Return minus fees and margin
    bt['ctrf'] = bt.ctr - bt.fee - bt.summargin
    
    # Daily Strategy Return
    bt['SR'] = np.where(bt.new_trade, bt.ctrf, bt.ctrf/bt.ctrf.shift(1))
    bt['DR'] = bt['close']/bt['close'].shift(1)

    # Adjust signal based on past performance
    # Best ASR: 0.989: 23.33 vs 18.60 
    # TODO: Set fee, ctr, ctrf, margin for Cash position
    bt['ASR'] = bt.SR.rolling(10).mean().shift(1)
    bt['signal'] = np.where(bt.ASR < 0.989, 'Cash', bt.signal)
    bt['SR'] = np.where(bt.signal == 'Cash', 1, bt.SR)

    bt['CSR'] = np.cumprod(bt.SR)
    bt['CMR'] = np.cumprod(bt.DR)

    return bt

def get_stats(ds):
    def my_agg(x):
        names = {
            'SRAvg': x['SR'].mean(),
            'SRTotal': x['SR'].prod(),
            'DRAvg': x['DR'].mean(),
            'DRTotal': x['DR'].prod(),
            'Count': x['y_pred_id'].count()
        }
    
        return pd.Series(names)

    stats = ds.groupby(ds['y_pred_id']).apply(my_agg)

    # Calculate Monthly Stats
    def my_agg(x):
        names = {
            'MR': x['DR'].prod(),
            'SR': x['SR'].prod()
        }
    
        return pd.Series(names)

    stats_mon = ds.groupby(ds['date'].map(lambda x: x.strftime('%Y-%m'))).apply(my_agg)
    stats_mon['CMR'] = np.cumprod(stats_mon['MR'])
    stats_mon['CSR'] = np.cumprod(stats_mon['SR'])
    
    return stats, stats_mon

def get_stats_mon(ds):
    def my_agg(x):
        names = {
            'MR': x['DR'].prod(),
            'SR': x['SR'].prod()
        }
    
        return pd.Series(names)
    
    return ds.groupby(ds['date'].map(lambda x: x.strftime('%m'))).apply(my_agg)    
    
def gen_trades(ds):
    def trade_agg(x):
        names = {
            'action': x.signal.iloc[0],    
            'open_ts': x.date.iloc[0],
            'close_ts': x.date_to.iloc[-1],
            'open': x.open.iloc[0],
            'close': x.close.iloc[-1],
            'duration': x.date.count(),
            'sl': x.sl.max(),
            'tp': x.tp.max(),
            'high': x.high.max(),
            'low': x.low.min(),
            'mr': x.DR.prod(),
            'sr': x.SR.prod()            
        }
    
        return pd.Series(names)

    tr = ds.groupby(ds.trade_id).apply(trade_agg)
#    if not p.short: tr = tr[tr.action=='Buy']
    tr['win'] = (tr.sr > 1) | ((tr.sr == 1) & (tr.mr < 1))
    tr['CMR'] = np.cumprod(tr['mr'])
    tr['CSR'] = np.cumprod(tr['sr'])
    tr = tr.dropna()
    
    return tr
    
def run_backtest(td, file):
    global stats
    global stats_mon
    global tr
    
    bt = run_pnl(td, file)

    bt['y_pred'] = td.y_pred
    bt['y_pred_val'] = td.y_pred_val
    bt['y_pred_id'] = td.y_pred_id
    bt['Price_Rise'] = np.where(bt['DR'] > 1, 1, 0)
    bt['date_to'] = bt['date'].shift(-1)
    bt.iloc[-1, bt.columns.get_loc('date_to')] = bt.iloc[-1, bt.columns.get_loc('date')] + dt.timedelta(minutes=p.trade_interval)

    stats, stats_mon = get_stats(bt)
#    bt = bt.merge(stats, left_on='y_pred_id', right_index=True, how='left')

    tr = gen_trades(bt)

    if p.charts: plot_chart(bt, file, 'date')
    if p.stats: show_stats(bt, tr)
    
    return bt

def plot_chart(df, title, date_col='date'):
    td = df.dropna().copy()
    td = td.set_index(date_col)
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    ax.plot(td['CSR'], color='g', label='Strategy Return')
    ax.plot(td['CMR'], color='r', label='Market Return')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

def show_stats(td, trades):
    avg_loss = 1 - trades[trades.win == False].sr.mean()
    avg_win = trades[trades.win].sr.mean() - 1
    r2r = 0 if avg_loss == 0 else avg_win / avg_loss
    win_ratio = len(trades[trades.win]) / len(trades)
    trade_freq = len(trades)/len(td)
    adr = trade_freq*(win_ratio * avg_win - (1 - win_ratio)*avg_loss)
    exp = 365 * adr
    sr = math.sqrt(365) * s.sharpe_ratio((td.SR - 1).mean(), td.SR - 1, 0)
    srt = math.sqrt(365) * s.sortino_ratio((td.SR - 1).mean(), td.SR - 1, 0)
    dur = trades.duration.mean()
    slf = len(td[td.sl])/len(td)
    tpf = len(td[td.tp])/len(td)
    print('Strategy Return: %.2f' % td.CSR.iloc[-1])
    print('Market Return: %.2f'   % td.CMR.iloc[-1])
    print('Sortino Ratio: %.2f' % srt)
    print('Bars in Trade: %.0f' % dur)
    print('Accuracy: %.2f' % (len(td[td.y_pred.astype('int') == td.Price_Rise])/len(td)))
    print('Win Ratio: %.2f' % win_ratio)
    print('Avg Win: %.2f' % avg_win)
    print('Avg Loss: %.2f' % avg_loss)
    print('Risk to Reward: %.2f' % r2r)
    print('Expectancy: %.2f' % exp)
    print('Sharpe Ratio: %.2f' % sr)
    print('Average Daily Return: %.3f' % adr)
    print('SL: %.2f TP: %.2f' % (slf, tpf))

# Inspired by:
# https://www.quantinsti.com/blog/artificial-neural-network-python-using-keras-predicting-stock-price-movement/
def runNN():
    global td
    global ds
    
    ds = dl.load_data()
    ds = add_features(ds)
    
    # Separate input from output. Exclude last row
    X = ds[p.feature_list][:-1]
#    y = ds[['DR']].shift(-1)[:-1]
    y = ds[['Price_Rise']].shift(-1)[:-1]

    # Split Train and Test and scale
    X_train, X_test, y_train, y_test = get_train_test(X, y)    
    
    K.clear_session() # Required to speed up model load
    if p.train:
        file = p.cfgdir+'/model.nn'
        nn = train_model(X_train, X_test, y_train, y_test, file)
    else:
        file = p.model
        nn = load_model(file) 
#        print('Loaded best model: '+file)
     
    # Making prediction
    y_pred_val = nn.predict(X_test)

    # Generating Signals
    td = gen_signal(ds, y_pred_val)

    # Backtesting
    td = run_backtest(td, file)

    print(str(get_signal_str()))

# See: 
# https://towardsdatascience.com/predicting-ethereum-prices-with-long-short-term-memory-lstm-2a5465d3fd
def runLSTM():
    global ds
    global td

    ds = dl.load_data()
    ds = add_features(ds)
   
    lag = 10
    n_features = 1
    X = pd.DataFrame()
    for i in range(1, lag+1):
        X['RSI'+str(i)] = ds['RSI'].shift(i)
#        X['MA'+str(i)] = ds['MA'].shift(i)
#        X['VOL'+str(i)] = ds['VOL'].shift(i)
    X = X.dropna()
    
    y = ds['DR']

    X_train, X_test, y_train, y_test = get_train_test(X, y) 

    X_train_t = X_train.reshape(X_train.shape[0], lag, n_features)
    X_test_t = X_test.reshape(X_test.shape[0], lag, n_features)

    file = p.model
    if p.train:
        file = p.cfgdir+'/model.nn'
        nn = Sequential()
        nn.add(LSTM(p.units, input_shape=(X_train_t.shape[1], X_train_t.shape[2]), return_sequences=True))
        nn.add(Dropout(0.2))
        nn.add(LSTM(p.units, return_sequences=False))
        nn.add(Dense(1))
        
        optimizer = RMSprop(lr=0.005, clipvalue=1.)
#        optimizer = 'adam'
        nn.compile(loss=p.loss, optimizer=optimizer)
        
        cp = ModelCheckpoint(file, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        h = nn.fit(X_train_t, y_train, batch_size = 10, epochs = p.epochs, 
                             verbose=0, callbacks=[cp], validation_data=(X_test_t, y_test))

        plot_fit_history(h)
    
    # Load Best Model
    nn = load_model(file)

    y_pred = nn.predict(X_test_t)    
    td = gen_signal(ds, y_pred)

    # Backtesting
    td = run_backtest(td, file)
    
    print(str(get_signal_str()))

def runModel(conf):
    p.load_config(conf)

    if p.model_type == 'NN':
        runNN()
    elif p.model_type == 'LSTM':
        runLSTM()
    
def check_retro():
    ret = (td.date >= '2019-03-05') & (td.date <= '2019-03-28')
    ret = ret | (td.date >= '2018-03-23') & (td.date <= '2018-04-15')
    ret = ret | (td.date >= '2018-07-26') & (td.date <= '2018-08-19')
    ret = ret | (td.date >= '2018-11-17') & (td.date <= '2018-12-06')
    ret = ret | (td.date >= '2017-12-03') & (td.date <= '2017-12-23')
    rtd = td[ret]
    print(rtd.SR.prod())
    print(rtd.DR.prod())
    print((len(rtd[rtd.y_pred.astype('int') == rtd.Price_Rise])/len(rtd)))

def check_missing_dates(td):
    from datetime import timedelta
    date_set = set(td.date.iloc[0] + timedelta(x) for x in range((td.date.iloc[-1] - td.date.iloc[0]).days))
    missing = sorted(date_set - set(td.date))
    print(missing)

# Tuning
#runModel('ETHBTCNN')
#runModel('ETHUSDNN1')

# PROD
#runModel('ETHUSDNN')
#runModel('BTCUSDNN')
