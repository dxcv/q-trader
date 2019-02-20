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
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation, Dropout
from sklearn.preprocessing import StandardScaler
#from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import keras.backend as K
import pandas as pd
import stats as s
import datalib as dl

def get_signal(offset=-1):
    s = td.iloc[offset]
    pnl = round(100*(s.ctrf - 1), 2)
    
    return {'new_trade':s.new_trade, 'action':s.signal, 
            'open':s.open_price, 'open_ts':s.date, 
            'close':s.close, 'close_ts':s.date_to, 'pnl':pnl, 'sl':s.sl, 'tp':s.tp}

def get_signal_str(s=''):
    if s == '': s = get_signal()
    txt = ''
    txt += 'NEW ' if s['new_trade'] else 'SAME '  
    txt += 'Trade: '+s['action'] 
    if p.short and s['action'] == 'Sell': txt += ' SHORT'
    txt += ' Open: '+str(s['open'])
    txt +=' Close: '+str(s['close'])
    txt +=' PnL: '+str(s['pnl'])+'%'
    if s['tp']: txt += ' TAKE PROFIT! '
    if s['sl']: txt += ' STOP LOSS! '
    
    return txt
 
def add_features(ds):
    print('*** Adding Features ***')
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
    train_split = int(len(ds)*p.train_pct)
    test_split = int(len(ds)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def plot_fit_history(h):
    # Plot model history
    # Accuracy: % of correct predictions 
    plt.plot(h.history['acc'], label='Train Accuracy')
    plt.plot(h.history['val_acc'], label='Test Accuracy')
    plt.plot(h.history['loss'], label='Train Loss')
    plt.plot(h.history['val_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(X_train, X_test, y_train, y_test, file):
    print('*** Training model with '+str(p.units)+' units per layer ***')
    nn = Sequential()
    nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu'))
    nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    cp = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    nn.compile(optimizer = 'adam', loss = p.loss, metrics = ['accuracy'])
    history = nn.fit(X_train, y_train, batch_size = 10, 
                             epochs = p.epochs, callbacks=[cp], 
                             validation_data=(X_test, y_test), 
                             verbose=0)

    # Plot model history
    plot_fit_history(history)

    # Load Best Model
    nn = load_model(file) 
    
    return nn
    
def gen_signal(ds, y_pred_val):
    print('*** Generating Signals ***')
    ds['y_pred_val'] = np.NaN
    ds.iloc[(len(ds) - len(y_pred_val)):,-1:] = y_pred_val
    ds['y_pred'] = (ds['y_pred_val'] >= p.signal_threshold)

    td = ds.dropna().copy()
    td['y_pred_id'] = np.trunc(td['y_pred_val'] * 100)
    td['signal'] = td['y_pred'].map({True: 'Buy', False: 'Sell'})
    if p.ignore_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.ignore_signals), np.NaN, td.signal)
        td['signal'] = td.signal.fillna(method='ffill')
    if p.hold_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.hold_signals), 'Hold', td.signal)

    return td

def run_pnl(td, file):
    bt = td[['date','open','high','low','close','signal']].copy()

    # TODO: Use Hold signal instead of Sell when Short is disabled
    # Calculate Min / Max Daily Return
    bt['minr'] = np.where(bt.signal == 'Buy', bt.low/bt.open, 1)
    if p.short:
        bt['minr'] = np.where(bt.signal == 'Sell', (2 - bt.high/bt.open), bt.minr)

    bt['maxr'] = np.where(bt.signal == 'Buy', bt.high/bt.open, 1)
    if p.short:
        bt['maxr'] = np.where(bt.signal == 'Sell', (2 - bt.low/bt.open), bt.maxr)

    # Calculate SL and TP
    bt['sl_price'] = np.where(bt.signal == 'Buy', bt.open * (1 - p.stop_loss), 0)
    if p.short:
        bt['sl_price'] = np.where(bt.signal == 'Sell', bt.open * (1 + p.stop_loss), bt.sl_price)
        
    bt['tp_price'] = np.where(bt.signal == 'Buy', bt.open * (1 + p.take_profit), 0)
    if p.short:
        bt['tp_price'] = np.where(bt.signal == 'Sell', bt.open * (1 - p.take_profit), bt.tp_price)

    bt['sl'] = (bt.signal == 'Buy') & (bt.low <= bt.sl_price) | p.short & (bt.signal == 'Sell') & (bt.high >= bt.sl_price)
    bt['tp'] = (bt.signal == 'Buy') & (bt.high >= bt.tp_price) | p.short & (bt.signal == 'Sell') & (bt.low <= bt.tp_price)

    bt['new_trade'] = (bt.signal != bt.signal.shift(1)) | bt.sl.shift(1) | bt.tp.shift(1)
    bt['trade_id'] = np.where(bt.new_trade, bt.index, np.NaN)
    bt['open_price'] = np.where(bt.new_trade, bt.open, np.NaN)
    bt = bt.fillna(method='ffill')

    # SL takes precedence over TP if both are happening in same timeframe
    bt['close_price'] = np.where(bt.tp, bt.tp_price, bt.close)
    bt['close_price'] = np.where(bt.sl, bt.sl_price, bt.close_price)   

    # Rolling Trade Return
    bt['ctr'] = np.where(bt.signal == 'Buy', bt.close_price/bt.open_price, 1)
    if p.short:
        bt['ctr'] = np.where(bt.signal == 'Sell', 2 - bt.close_price/bt.open_price, bt.ctr)

    # Margin Calculation. Assuming marging is used for short trades only
    bt['margin'] = 0
    if p.short:
        bt['margin'] = np.where(bt['signal'] == 'Sell',  p.margin, bt.margin)
        bt['margin'] = np.where(bt.new_trade & (bt['signal'] == 'Sell'),  p.margin + p.margin_open, bt.margin)
    
    bt['summargin'] = bt.groupby('trade_id')['margin'].transform(pd.Series.cumsum)

    # Rolling Trade Open and Close Fees
    bt['fee'] = np.where((bt.signal == 'Buy') | (p.short & (bt.signal == 'Sell')), p.fee + bt.ctr*p.fee, 0)
    
    # Rolling Trade Return minus fees and margin
    bt['ctrf'] = bt.ctr - bt.fee - bt.summargin
    
    # Daily Strategy Return
    bt['SR'] = np.where(bt.new_trade, bt.ctrf, bt.ctrf/bt.ctrf.shift(1))
    bt['DR'] = bt['close']/bt['close'].shift(1)
    bt['CSR'] = np.cumprod(bt.SR)
    bt['CMR'] = np.cumprod(bt.DR)
    
    return bt

def get_stats(ds):
    def my_agg(x):
        names = {
            'SRAvg': x['SR'].mean(),
            'SRTotal': x['SR'].prod(),
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
    tr['win'] = (tr.sr > 1)
    tr['CMR'] = np.cumprod(tr['mr'])
    tr['CSR'] = np.cumprod(tr['sr'])
    tr = tr.dropna()
    if not p.short: tr = tr[tr.action == 'Buy']
    
    return tr
    
def run_backtest(td, file):
    global stats
    global stats_mon
    global tr
    
    bt = run_pnl(td, file)

    bt['y_pred'] = td.y_pred
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
    td = df.copy()
    if p.plot_bars > 0:
        td = td[td[date_col] >= td[date_col].max() - dt.timedelta(days=p.plot_bars)]
#        td = td.tail(p.plot_bars).reset_index(drop=True)
        td['CMR'] = dl.normalize(td['CMR'])
        td['CSR'] = dl.normalize(td['CSR'])
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
    avg_loss = 1 - trades[trades.sr < 1].sr.mean()
    avg_win = trades[trades.sr > 1].sr.mean() - 1
    r2r = avg_win / avg_loss
    win_ratio = len(trades[trades.win]) / len(trades)
    trade_freq = len(trades[trades.sr != 1])/len(td)
    adr = trade_freq*(win_ratio * avg_win - (1 - win_ratio)*avg_loss)
    exp = 365 * adr
    rar = exp / (100 * p.stop_loss)
    sr = math.sqrt(365) * s.sharpe_ratio((td.SR - 1).mean(), td.SR - 1, 0)
    srt = math.sqrt(365) * s.sortino_ratio((td.SR - 1).mean(), td.SR - 1, 0)
    dur = trades.duration.mean()
    false_stop = len(td[(td.y_pred.astype('int') == td.Price_Rise) & td.sl])/(len(td[td.sl]) + 0.01)
    print('Strategy Return: %.2f' % trades.CSR.iloc[-1])
    print('Market Return: %.2f'   % trades.CMR.iloc[-1])
    print('Sortino Ratio: %.2f' % srt)
    print('Bars in Trade: %.0f' % dur)
    print('Accuracy: %.2f' % (len(td[td.y_pred.astype('int') == td.Price_Rise])/len(td)))
    print('Win Ratio: %.2f' % win_ratio)
    print('Avg Win: %.2f' % avg_win)
    print('Avg Loss: %.2f' % avg_loss)
    print('Risk to Reward: %.2f' % r2r)
    print('Expectancy: %.2f' % exp)
    print('Risk Adjusted Return: %.2f' % rar)
    print('Sharpe Ratio: %.2f' % sr)
    print('Average Daily Return: %.3f' % adr)
    print('False Stops: %.2f' % false_stop)

# Inspired by:
# https://www.quantinsti.com/blog/artificial-neural-network-python-using-keras-predicting-stock-price-movement/
def runNN():
    global td
    global ds
    
    ds = dl.load_price_data()
    ds = add_features(ds)
    
    # Separate input from output. Exclude last row
    X = ds[p.feature_list][:-1]
    y = ds[['Price_Rise']].shift(-1)[:-1]

    # Split Train and Test and scale
    X_train, X_test, y_train, y_test = get_train_test(X, y)    
    
    if p.train:
        file = p.cfgdir+'/model.nn'
        nn = train_model(X_train, X_test, y_train, y_test, file)
    else:
        file = p.model
        nn = load_model(file) 
        print('Loaded best model: '+file)
     
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

    ds = dl.load_price_data()

    # Add features
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['DR'] = ds['close']/ds['close'].shift(1)
   
    lag = 10
    for i in range(1, lag+1):
        ds['RSI'+str(i)] = ds['RSI'].shift(i)
    ds = ds.dropna()
    
    X = ds.iloc[:,-lag:]
    y = ds['DR']

    X_train, X_test, y_train, y_test = get_train_test(X, y) 

    X_train_t = X_train.reshape(X_train.shape[0], 1, lag)
    X_test_t = X_test.reshape(X_test.shape[0], 1, lag)

    file = p.model
    if p.train:    
        file = p.cfgdir+'/model.lstm'
        K.clear_session()
        nn = Sequential()
        nn.add(LSTM(p.units, input_shape=(1, lag), return_sequences=True))
        nn.add(Dropout(0.2))

        nn.add(LSTM(p.units, return_sequences=False))
        nn.add(Dropout(0.2))

        nn.add(Dense(1))
        
        optimizer = RMSprop(lr=0.005, clipvalue=1.)
#        optimizer = 'adam'
        nn.compile(loss=p.loss, optimizer=optimizer, metrics = ['accuracy'])
        
        cp = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        h = nn.fit(X_train_t, y_train, batch_size = 10, epochs = p.epochs, 
                             verbose=1, callbacks=[cp], validation_data=(X_test_t, y_test))
        
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

#runModel('BTCUSDNN')

#runModel('ETHUSDNN')
#runModel('ETHUSDLSTM')
