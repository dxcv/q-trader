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
import qlib as q
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
import stats as s

def get_signal():
    new = td.new.iloc[-1]
    start = td.date_to.iloc[-1]
    end = start + dt.timedelta(minutes=p.trade_interval)
    
    return {'new': new, 'signal': td.signal.iloc[-1], 'start': start, 'end': end}


# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', init='uniform'):
    print('Using NN with '+str(p.units)+' units per layer')
    model = Sequential()
    model.add(Dense(p.units, kernel_initializer = init, activation = 'relu', input_dim = X.shape[1]))
    model.add(Dense(p.units, kernel_initializer = init, activation = 'relu'))
    model.add(Dense(1, kernel_initializer = init, activation = 'sigmoid'))

	# Compile model
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def runNN1(conf):
    #from pandas import read_csv, set_option
    from sklearn.preprocessing import StandardScaler
#    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import accuracy_score
#    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from keras.wrappers.scikit_learn import KerasClassifier

    global ds
    global grid_result

    seed = 7
    np.random.seed(seed)

    q.init(conf)
    ds = q.load_data()    

    #  Most used indicators: https://www.quantinsti.com/blog/indicators-build-trend-following-strategy/
    ds['date_to'] = ds['date'].shift(-1)
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
    
    # Tomorrow Return - this should not be included in training set
    ds['TR'] = ds['DR'].shift(-1)
    # Predicted value is whether price will rise
    ds['Price_Rise'] = np.where(ds['TR'] > 1, 1, 0)
    
    if p.max_bars > 0: ds = ds.tail(p.max_bars).reset_index(drop=True)
    ds = ds.dropna()

    # Separate input from output
    X = ds.iloc[:, -11:-2]
    y = ds.iloc[:, -1]
    
    # Separate train from test
    train_split = int(len(ds)*p.train_pct)
    test_split = int(len(ds)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Cross Validation
    model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=5)
#    estimators = []
#    estimators.append(('standardize', StandardScaler()))
#    estimators.append(('mlp', model))
#    pipeline = Pipeline(estimators)
#    results = cross_val_score(pipeline, X_train, y_train, cv=TimeSeriesSplit(n_splits=5))
#    print(results.mean())

    
    # Grid search epochs, batch size and optimizer
    # See: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#    optimizers = ['rmsprop', 'adam']
#    initial = ['glorot_uniform', 'normal', 'uniform']
#    epochs = [100]
#    batches = [10]
#    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=initial)
#    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=TimeSeriesSplit())
#    grid_result = grid.fit(X_train, y_train)

    # summarize results
#    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#    means = grid_result.cv_results_['mean_test_score']
#    stds = grid_result.cv_results_['std_test_score']
#    params = grid_result.cv_results_['params']
#    for mean, stdev, param in zip(means, stds, params):
#        print("%f (%f) with: %r" % (mean, stdev, param))

    models = []
    models.append(('NN', model))
    models.append(('LR' , LogisticRegression()))
    models.append(('LDA' , LinearDiscriminantAnalysis()))
    models.append(('KNN' , KNeighborsClassifier()))
    models.append(('CART' , DecisionTreeClassifier()))
    models.append(('NB' , GaussianNB()))
    models.append(('SVM' , SVC()))
    models.append(('RF' , RandomForestClassifier(n_estimators=50)))
    models.append(('XGBoost', XGBClassifier()))

    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred = y_pred.round()
        accu_score = accuracy_score(y_test, y_pred)
        print(name + ": " + str(accu_score))
 
def plot_chart(td, title):
    if p.plot_bars > 0 and not p.train: 
        td = td.tail(p.plot_bars).reset_index(drop=True)
        td['CMR'] = q.normalize(td['CMR'])
        td['CSR'] = q.normalize(td['CSR'])
    td = td.set_index('date')
    fig, ax = plt.subplots()
    # fig.autofmt_xdate()
    ax.plot(td['CSR'], color='g', label='Strategy Return')
    ax.plot(td['CMR'], color='r', label='Market Return')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

def show_stats():
    avg_loss = 1 - trades[trades.SR1 < 1].SR1.mean()
    avg_win = trades[trades.SR1 >= 1].SR1.mean() - 1
    win_ratio = len(trades[trades.SR1 >= 1]) / len(trades)
    trade_freq = len(trades) / (trades.open_ts.max() - trades.open_ts.min()).days
    exp = 365 * trade_freq * (win_ratio * avg_win - (1 - win_ratio)*avg_loss)
    sr = s.sharpe_ratio((trades.SR1 - 1).mean(), trades.SR1 - 1, 0)
    print('Strategy Return: %.2f' % td.CSR.iloc[-1])
    print('Market Return: %.2f'   % td.CMR.iloc[-1])
    print('Trade Frequency: %.2f' % trade_freq)
    print('Accuracy: %.2f' % (len(td[td.y_pred.astype('int') == td.Price_Rise])/len(td)))
    print('Win Ratio: %.2f' % win_ratio)
    print('Avg Win: %.2f' % avg_win)
    print('Avg Loss: %.2f' % avg_loss)
    print('Expectancy: %.2f' % exp)
    print('Sharpe Ratio: %.2f' % sr)
    print('Average Daily Return: %.3f' % np.mean(td.SR - 1))
    

# Source:
# https://www.quantinsti.com/blog/artificial-neural-network-python-using-keras-predicting-stock-price-movement/
def runNN(conf):
    global td
    global ds
    global X
    global stats
    global stats_mon
    global trades
    
    q.init(conf)
    ds = q.load_data()
#    ds = load_prices()
    
    #  Most used indicators: https://www.quantinsti.com/blog/indicators-build-trend-following-strategy/
    ds['date_to'] = ds['date'].shift(-1)
    # Calculate Features
    ds['VOL'] = ds['volumeto']/ds['volumeto'].rolling(window = p.vol_period).mean()
    ds['HH'] = ds['high']/ds['high'].rolling(window = p.hh_period).max() 
    ds['LL'] = ds['low']/ds['low'].rolling(window = p.ll_period).min()
#    ds['HHLL'] = (ds.high+ds.low)/(ds.high/ds.HH+ds.low/ds.LL)
    ds['DR'] = ds['close']/ds['close'].shift(1)
    ds['MA'] = ds['close']/ds['close'].rolling(window = p.sma_period).mean()
    ds['MA2'] = ds['close']/ds['close'].rolling(window = 2*p.sma_period).mean()
    ds['Std_dev']= ds['close'].rolling(p.std_period).std()/ds['close']
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['Williams %R'] = talib.WILLR(ds['high'].values, ds['low'].values, ds['close'].values, p.wil_period)
    
    # Tomorrow Return - this should not be included in training set
    ds['TR'] = ds['DR'].shift(-1)
    # Predicted value is whether price will rise
    ds['Price_Rise'] = np.where(ds['TR'] > 1, 1, 0)
    
    if p.max_bars > 0: ds = ds.tail(p.max_bars).reset_index(drop=True)
    ds = ds.dropna()

    # Separate input from output
    X = ds[['VOL','HH','LL','DR','MA','MA2','Std_dev','RSI','Williams %R']]
    y = ds[['Price_Rise']]
    
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
        cp = ModelCheckpoint(model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
        history = classifier.fit(X_train, y_train, batch_size = 10, epochs = p.epochs, callbacks=[cp], validation_data=(X_test, y_test), verbose=0)
    
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
    td['new'] = td['signal'] != td['signal'].shift(1)
    # Generate Trade List
    td['action'] = td['signal'].shift(1)
    td['trade_id'] = np.where(td.new.shift(1), td.index, np.NaN)
    td['trade_id'] = td.trade_id.fillna(method='ffill')

    def trade_agg(x):
        names = {
            'action': x.action.iloc[0],    
            'open_ts': x.date.iloc[0],
            'close_ts': x.date_to.iloc[-1],
            'open_price': x.open.iloc[0],
            'close_price': x.close.iloc[-1]            
        }
    
        return pd.Series(names)

    trades = td.groupby(td.trade_id).apply(trade_agg)
    trades['hours'] = (trades.close_ts - trades.open_ts).astype('timedelta64[h]')
    trades['margin'] = np.where(p.short and trades.action == 'Sell', trades.hours/24 * p.margin, 0)
    trades['MR'] = trades['close_price']/trades['open_price']
    trades['SR'] = np.where(trades['action'] == 'Buy', trades['MR'], np.NaN)
    trades['SR'] = np.where(trades['action'] == 'Sell', (2 - trades['MR']) if p.short else 1, trades.SR)
    trades['SR'] = np.where(trades['action'] == 'Cash', 1, trades.SR)
    # Fee is applied twice: on open and close position
    trades['SR1'] = trades['SR'] * (1 - p.fee)**2 * (1 - trades.margin)
    trades['CMR'] = np.cumprod(trades['MR'])
    trades['CSR'] = np.cumprod(trades['SR1'])
    
    td['fee'] = np.where(td.new, (1 - p.fee)**(2 if p.short else 1), 1)
    td['fee'] = np.where(td.new & (td.signal == 'Cash'), 1 - p.fee, td.fee)
    td['margin'] = np.where(p.short and td['signal'] == 'Sell',  1 - p.margin, 1)
    td['SR'] = np.where(td['signal'] == 'Buy', td['TR'], np.NaN)
    td['SR'] = np.where(td['signal'] == 'Sell', (2 - td['TR']) if p.short else 1, td.SR)
    td['SR'] = np.where(td['signal'] == 'Cash', 1, td.SR)
    td['SR'] = td['SR'] * td['fee'] * td['margin']
    td['CMR'] = np.cumprod(td['TR'])
    td['CSR'] = np.cumprod(td['SR'])
    
    def my_agg(x):
        names = {
            'SRAvg': x['SR'].mean(),
            'SRTotal': x['SR'].prod(),
            'Price_Rise_Prob': x['Price_Rise'].mean(),
            'YPredCount': x['TR'].count()
        }
    
        return pd.Series(names)

    stats = td.groupby(td['y_pred_id']).apply(my_agg)
    td = td.merge(stats, left_on='y_pred_id', right_index=True, how='left')

    # Calculate Monthly Stats
    def my_agg(x):
        names = {
            'MR': x['TR'].prod(),
            'SR': x['SR'].prod()
        }
    
        return pd.Series(names)

    stats_mon = td.groupby(td['date'].map(lambda x: x.strftime('%Y-%m'))).apply(my_agg)
    stats_mon['CMR'] = np.cumprod(stats_mon['MR'])
    stats_mon['CSR'] = np.cumprod(stats_mon['SR'])
    
    if p.charts: plot_chart(td, model)
    if p.stats: show_stats()

    print(str(get_signal()))

#runNN('ETHUSDNN')
