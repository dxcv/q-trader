#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:28:17 2018

@author: igor
"""

import datetime as dt
import params as p
import talib
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import datalib as dl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
import stats as s

def get_signal(offset=-1):
    s = td.iloc[offset]
    pnl = round(100*(s.SR - 1), 2)
    
    return {'new':s.new, 'action':s.signal, 'open':s.open, 'open_ts':s.date, 
            'close':s.close, 'close_ts':s.date_to, 'pnl':pnl, 'sl':s.sl, 'tp':s.tp}

def get_signal_str(offset=-1):
    s = get_signal(offset)
    
    txt = ''
    if s['tp']: txt = '!!!TAKE PROFIT!!! '
    if s['sl']: txt = '!!!STOP LOSS!!! '
    if s['new']: txt += 'NEW SIGNAL! '  
    txt += 'Action: '+s['action']
    txt += ' Open: '+str(s['open'])
    txt +=' Close: '+str(s['close'])
    txt +=' PnL: '+str(s['pnl'])+'%'
    
    return txt
 
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

# Source:
# https://www.quantinsti.com/blog/artificial-neural-network-python-using-keras-predicting-stock-price-movement/
def runNN(conf):
    global td
    global ds
    global X
    global stats
    global stats_mon
    global trades
    
    p.load_config(conf)
    ds = dl.load_data()
#    ds = q.load_prices()
    
    #  Most used indicators: https://www.quantinsti.com/blog/indicators-build-trend-following-strategy/
    ds['date_to'] = ds['date'].shift(-1)
    # Set date_to to next date
    ds.iloc[-1, ds.columns.get_loc('date_to')] = ds.iloc[-1, ds.columns.get_loc('date')] + dt.timedelta(minutes=p.trade_interval)
    # Calculate Features
    ds['VOL'] = ds['volumeto']/ds['volumeto'].rolling(window = p.vol_period).mean()
    ds['HH'] = ds['high']/ds['high'].rolling(window = p.hh_period).max() 
    ds['LL'] = ds['low']/ds['low'].rolling(window = p.ll_period).min()
    ds['DR'] = ds['close']/ds['close'].shift(1)
    ds['MA'] = ds['close']/ds['close'].rolling(window = p.sma_period).mean()
    ds['MA2'] = ds['close']/ds['close'].rolling(window = 2*p.sma_period).mean()
    ds['Std_dev']= ds['close'].rolling(p.std_period).std()/ds['close']
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['Williams %R'] = talib.WILLR(ds['high'].values, ds['low'].values, ds['close'].values, p.wil_period)
    ds['Price_Rise'] = np.where(ds['DR'] > 1, 1, 0)
    ds = ds.dropna()
    
    # Separate input from output. Exclude last row
    X = ds[['VOL','HH','LL','DR','MA','MA2','Std_dev','RSI','Williams %R']][:-1]
    y = ds[['Price_Rise']].shift(-1)[:-1]
    
    # Separate train from test
    train_split = int(len(ds)*p.train_pct)
    test_split = int(len(ds)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Building Neural Network
    print('Using NN with '+str(p.units)+' units per layer')
    classifier = Sequential()
    classifier.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
    classifier.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    if p.train:
        # Early stopping  
        #es = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='max')
        model = p.cfgdir+'/model.nn'
        cp = ModelCheckpoint(model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
        history = classifier.fit(X_train, y_train, batch_size = 10, 
                                 epochs = p.epochs, callbacks=[cp], 
                                 validation_data=(X_test, y_test), 
                                 verbose=0)
    
        # Plot model history
        # Accuracy: % of correct predictions 
        plt.plot(history.history['acc'], label='Train Accuracy')
        plt.plot(history.history['val_acc'], label='Test Accuracy')
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        model = p.model
    
    # Load Best Model
    classifier.load_weights(model)
    print('Loaded Best Model From: '+model)
    
    # Compile model (required to make predictions)
    classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
    
    # Predicting The Price
    y_pred_val = classifier.predict(X_test)

    ds['y_pred_val'] = np.NaN
    ds.iloc[(len(ds) - len(y_pred_val)):,-1:] = y_pred_val
    ds['y_pred'] = (ds['y_pred_val'] >= 0.5)

    td = ds.dropna().copy()
    td['y_pred_id'] = np.trunc(td['y_pred_val'] * 10)
    td['signal'] = td['y_pred'].map({True: 'Buy', False: 'Sell'})
    if p.ignore_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.ignore_signals), np.NaN, td.signal)
        td['signal'] = td.signal.fillna(method='ffill')
    td['minr'] = np.where(td.signal == 'Buy', td.low / td.open, np.NaN)
    td['minr'] = np.where(td.signal == 'Sell', (2 - td.high / td.open) if p.short else 1, td.minr)
    td['sl'] = td.minr < 1 - p.stop_loss
    td['maxr'] = np.where(td.signal == 'Buy', td.high / td.open, np.NaN)
    td['maxr'] = np.where(td.signal == 'Sell', (2 - td.low / td.open) if p.short else 1, td.maxr)
    td['tp'] = td.maxr > 1 + p.take_profit
    # New trade if signal changes or SL/TP was triggered before
    td['new'] = np.where(td.sl.shift(1) | td.tp.shift(1), True, False)  
    td['new'] = np.where(td.signal != td.signal.shift(1), True, td.new)
    # Add open fee for each new trade
    td['open_fee'] = np.where(td.new, 1 - p.fee, 1)
    td['close_fee'] = np.where(td.new.shift(-1), 1 - p.fee, 1)
    if not p.short:
        td['open_fee'] = np.where(td.new & (td.signal == 'Sell'), 1, td.open_fee)
        td['close_fee'] = np.where(td.new.shift(-1) & (td.signal == 'Sell'), 1, td.close_fee)
    td['margin'] = np.where(p.short and td['signal'] == 'Sell',  1 - p.margin, 1)
    td['SR'] = np.where(td['signal'] == 'Buy', td['DR'], np.NaN)
    td['SR'] = np.where(td['signal'] == 'Sell', (2 - td['DR']) if p.short else 1, td.SR)
    # FIXME: When SL and TP happen for same trade - take SL. But this should be based on actual timing
    td['SR'] = np.where(td.tp, 1 + p.take_profit, td.SR)
    td['SR'] = np.where(td.sl, 1 - p.stop_loss, td.SR)
    td['SR'] = td['SR'] * td['open_fee'] * td['close_fee'] * td['margin']
    td['CMR'] = np.cumprod(td['DR'])
    td['CSR'] = np.cumprod(td['SR'])    

    def my_agg(x):
        names = {
            'SRAvg': x['SR'].mean(),
            'SRTotal': x['SR'].prod(),
            'Price_Rise_Prob': x['Price_Rise'].mean(),
            'YPredCount': x['y_pred_id'].count()
        }
    
        return pd.Series(names)

    stats = td.groupby(td['y_pred_id']).apply(my_agg)
    td = td.merge(stats, left_on='y_pred_id', right_index=True, how='left')

    # Calculate Monthly Stats
    def my_agg(x):
        names = {
            'MR': x['DR'].prod(),
            'SR': x['SR'].prod()
        }
    
        return pd.Series(names)

    stats_mon = td.groupby(td['date'].map(lambda x: x.strftime('%Y-%m'))).apply(my_agg)
    stats_mon['CMR'] = np.cumprod(stats_mon['MR'])
    stats_mon['CSR'] = np.cumprod(stats_mon['SR'])
    
    # Generate Trade List
    td['trade_id'] = np.where(td.new, td.index, np.NaN)
    td = td.fillna(method='ffill')

    def trade_agg(x):
        names = {
            'action': x.signal.iloc[0],    
            'open_ts': x.date.iloc[0],
            'close_ts': x.date_to.iloc[-1],
            'open': x.open.iloc[0],
            'close': x.close.iloc[-1],
            'sl': x.sl.max(),
            'tp': x.tp.max(),
            'high': x.high.max(),
            'low': x.low.min(),
            'margin': x.margin.prod(),
            'mr': x.DR.prod(),
            'sr': x.SR.prod()            
        }
    
        return pd.Series(names)

    trades = td.groupby(td.trade_id).apply(trade_agg)
    trades['cmr'] = np.cumprod(trades['mr'])
    trades['csr'] = np.cumprod(trades['sr'])
    trades = trades.dropna()
    
    if p.charts: plot_chart(td, model, 'date')
    
    if p.stats:
        avg_loss = 1 - td[td.SR < 1].SR.mean()
        avg_win = td[td.SR >= 1].SR.mean() - 1
        r2r = avg_win / avg_loss
        win_ratio = len(td[td.SR >= 1]) / len(td)
#        trade_freq = len(trades) / (trades.close_ts.max() - trades.open_ts.min()).days
        trade_freq = 1
        adr = trade_freq * (win_ratio * avg_win - (1 - win_ratio)*avg_loss)
        exp = 365 * adr
        rar = exp / (100 * p.stop_loss)
        sr = s.sharpe_ratio((td.SR - 1).mean(), td.SR - 1, 0)
        print('Strategy Return: %.2f' % td.CSR.iloc[-1])
        print('Market Return: %.2f'   % td.CMR.iloc[-1])
        print('Trade Frequency: %.2f' % trade_freq)
        print('Accuracy: %.2f' % (len(td[td.y_pred.astype('int') == td.Price_Rise])/len(td)))
        print('Win Ratio: %.2f' % win_ratio)
        print('Avg Win: %.2f' % avg_win)
        print('Avg Loss: %.2f' % avg_loss)
        print('Risk to Reward: %.2f' % r2r)
        print('Stop Loss: %.2f' % p.stop_loss)
        print('Take Profit: %.2f' % p.take_profit)
        print('Expectancy: %.2f' % exp)
        print('Risk Adjusted Return: %.2f' % rar)
        print('Sharpe Ratio: %.2f' % sr)
        print('Average Daily Return: %.3f' % adr)
        

    print(str(get_signal_str()))

#runNN('BTCUSDNN')
#runNN('ETHUSDNN')
#runNN('XRPUSDNN')
#runNN('XMRUSDNN')
#runNN('ETCUSDNN')
