#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:40:08 2017

@author: imonahov
"""

import math

def truncate(n, digits):
    return math.trunc(n*(10**digits))/(10**digits)

def load_config(config):
    global conf
    conf = config
    global random_scale
    random_scale = 0.00001 # Defines standard deviation for random Q values 
    global start_balance
    start_balance = 1.0
    global short
    short = False # Short trading
    global actions
    actions = 2 # Number of actions (% of stock holdings) 2 for long only, 3 to add short
    # α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
    global alpha
    alpha = 0.2
    # γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards. Best: 0.9
    global gamma
    gamma = 0.9
    # Probability to chose random action instead of best action from Q Table. Best values: 0.2 - 0.5
    global epsilon
    epsilon = 0.5
    global train
    train = False # Train model
    global reload
    reload = False # Force to reload price data. False means reload only if data is old 
    global charts
    charts = True # Plot charts
    global stats
    stats = True # Show model stats
    global epochs
    epochs = 30 # Number of iterations for training (best 50)
    global features
    features = 4 # Number of features in state for Q table
    global feature_bins
    feature_bins = 3 # Number of bins for feature (more bins tend to overfit)
    global max_r
    max_r = 0
    global ticker
    ticker = conf[0:3]
    global currency
    currency = conf[3:6]
    global pair # Used for Exchange
    pair = ticker+'/'+currency
    global cfgdir
    cfgdir = 'data/'+conf
    global version
    version = 2 # Model version
    global sma_period
    sma_period = 50 # Best: 50 Alternative: 21, 55, 89, 144, 233 / 50, 100, 200
    global adr_period
    adr_period = 20 # Average Daily Return period
    global hh_period
    hh_period = 50 # Window for Highest High (best: 20 - 50)
    global ll_period
    ll_period = 50 # Window for Lowest Low (best: 20 - 50)
    global rsi_period
    rsi_period = 50
    global vol_period
    vol_period = 30
    global std_period
    std_period = 7
    global wil_period
    wil_period = 7
    global adx_period
    adx_period = 6
    global exchange
    exchange = 'CCCAGG' # Average price from all exchanges
#    exchange = 'KRAKEN'
    global execute
    execute = False
    global order_size # Order size in equity. 0 means to use order_pct
    order_size = 0
    global max_short # Max Order size for short position
    max_short = 0
    global order_pct # % of balance to use for position
    order_pct = 1
    global order_precision # Number of digits after decimal for order size
    order_precision = 2
    global result_size
    result_size = 0
    global order_wait # Wait time in seconds for order to be filled
    order_wait = 5*60
    global order_type # Order Type: market or limit 
    order_type = 'limit'
    global min_cash
    min_cash = 1
    global min_equity # Minimum order size
    min_equity = 0.001
    global bar_period
    bar_period = 'day' # Price bar period: day or hour
    global max_bars
    max_bars = 0 # Number of bars to use for training
    global train_goal
    train_goal = 'R' # Maximize Return
    global limit_fee # Exchange fee
    limit_fee = 0.0010 # Kraken Maker fee
    global market_fee # Kraken Taker fee
    market_fee = 0.0016
    global margin # Daily Margin fee for short positions
    margin = 0.0012 # Kraken 24 margin fee
    global margin_open # Kraken Margin Open fee
    margin_open = 0.0002
    global ratio
    ratio = 0 # Min ratio for Q table to take an action
    global units
    units = 16
    global train_pct
    train_pct = 0.8 # % of data used for training
    global test_pct
    test_pct = 0.2 # % of data used for testing
    global model
    model = cfgdir+'/model.nn'
    global test_bars # Number of bars to test. Overrides test_pct if > 0
    test_bars = 0
    global time_lag # Number of hours to offset price data. 0 means no offset
    time_lag = 0 # best 0: 3.49 4: 2.59 6: 1.6 7: 1.49 8: 2.71 20: 0.87
    global trade_interval
    trade_interval = 60*24 # Trade interval in minutes
    global sleep_interval
    sleep_interval = 60 # Bot sleep interval in seconds when waiting for new signal / network error
    global ignore_signals
    ignore_signals = None # list of y_pred_id to ignore. None to disable 
    global hold_signals # list of y_pred_id to HOLD. None to disable
    hold_signals = None
    global min_data_size # Minimum records expected from Cryptocompare API
    min_data_size = 100
    global take_profit # Take Profit % Default 1 which is no TP
    take_profit = 1
    global stop_loss
    stop_loss = 0
    global buy_sl # Enables SL for Buy
    buy_sl = False # Buy SL is disabled as not profitable
    global sell_sl # Enables SL for Sell
    sell_sl = False
    global buy_tp # Enables TP for Buy
    buy_tp = False
    global sell_tp # Enables TP for Sell
    sell_tp = False # Sell TP is disabled as cannot have both SL and TP on Kraken
    global leverage # Leverage used for margin trading. 0 means - no leverage
    leverage = 2
    global feature_list # List of features to use for NN (ordered by importance)
    feature_list = ['VOL','HH','LL','DR','MA','MA2','STD','RSI','WR','DMA','MAR'] 
#    features ordered by importance: ['RSI','MA','MA2','STD','WR','MAR','HH','VOL','LL','DMA','DR']
    global datasource # Data Source for price data. Options cc: CryptoCompare, kr: Kraken, dr: DataReader, ql: Quandl
    datasource = 'cc'
    global loss # Loss function for NN: mse, binary_crossentropy, mean_absolute_error etc
    loss = 'mse'
    global signal_threshold
    signal_threshold = 0.5
    global model_type # Model Type to run: NN, LSTM
    model_type = 'runNN'
    global price_precision # Number of decimals for price
    price_precision = 2
    global kraken_pair # Name of pair in Kraken. Used for fetching price data from Kraken
    kraken_pair = ''
    global breakout # Use Breakout strategy
    breakout = False
    global signal_scale # Used for signal grouping by y_pred_id 
    signal_scale = 1000
    global adjust_signal
    adjust_signal = True
    global batch_size
    batch_size = 0
    global btc_data # Use asset price in BTC for model
    btc_data = False

    if conf == 'BTCUSD': # R: 180.23 SR: 0.180 QL/BH R: 6.79 QL/BH SR: 1.80
#        train = True
        max_r = 18
        version = 1
    elif conf == 'ETHUSD': # R: 6984.42 SR: 0.164 QL/BH R: 8.94 QL/BH SR: 1.30
#        6508 / 1.25
        max_r = 6508
#        train = True
#        epsilon = 0
    elif conf == 'NVDA':
        datasource = 'ql'
#        reload = True
#        test_pct = 1
        units = 32
        train = True
        take_profit = 1
        epochs = 30
        feature_list = ['MA','MA2','RSI','HH','LL','DMA','MAR','VOL']
        sma_period = 50
        rsi_period = 21
        hh_period = 50
        ll_period = 50
        vol_period = 50
    elif conf == 'BTCUSDLSTM':
#        model = cfgdir+'/model.top'
#        model = 'data/ETHUSDLSTM/model.nn'
        model_type = 'LSTM'
        signal_threshold = 1
#        short = True
        train = True
        train_pct = 1
        test_pct = 1
        units = 32
        epochs = 20
        limit_fee = 0.002 # Taker
        order_type = 'market'
        rsi_period = 50
    elif conf == 'ETHUSDLSTM':
# Accuracy: 0.57, Win Ratio: 0.68, Strategy Return: 1.77
#        train = True
#        train_pct = 1
#        test_pct = 1
#        test_bars = 365
#        test_pct = 1
        units = 32
        epochs = 20
        model_type = 'LSTM'
        signal_threshold = 1
        model = cfgdir+'/model.top'
        take_profit = 0.15  # Best TP 0.15: 1.77 No: 1.45
#        execute = True
        limit_fee = 0.002 # Taker
        order_type = 'market'
        order_pct = 0.99 # Reserve 1% for slippage
#        !!! Short only in Bear market !!!
#        short = True
#        max_short = 250
    elif conf == 'ETHUSDLSTM1':
        train = True
        train_pct = 0.7
        test_pct = 0.3
#        test_pct = 1
        model_type = 'LSTM'
        units = 20
        epochs = 20
        signal_threshold = 1
    elif conf == 'ETHBTCNN':
        feature_list = ['MA','MA2']
        exchange = 'KRAKEN'
        datasource = 'kr'
        kraken_pair = 'XETHXXBT'
        reload = True
#        train = True
#        test_bars = 272
#        test_pct = 1
        model = cfgdir+'/model.top'
        units = 20
        epochs = 20
#        breakout = True
#        sell_sl = True
        limit_fee = 0.0008 # Maker
#        short = True
    elif conf == 'BTCUSDNN':
#        execute = True
        breakout = True
        order_pct = 1
        short = True
        datasource = 'kr'
        exchange = 'KRAKEN'
        kraken_pair = 'XXBTZUSD'
        reload = True
#        train = True
#        test_pct = 1
        test_bars = 365
        units = 20
        epochs = 30
        model = cfgdir+'/model.top'
        limit_fee = 0.0008 # Maker
    elif conf == 'ETHUSDNN2':
# model.599: 23 Nov SR: 5.99 Kraken: 194186 (epoch ± 600) train 0.75 test 0.25, no buy SL 
        breakout = True
        sell_sl = True
#        buy_sl = True
        short = True
        leverage = 5
        min_equity = 0.02
        order_precision = 0
        exchange = 'KRAKEN'
        datasource = 'kr'
        kraken_pair = 'XETHZUSD'
        reload = True
#        train = True
        train_pct = 0.75
#        test_pct = 0.25
        test_pct = 1
#        test_bars = 365
        units = 32
        epochs = 1000
        model = cfgdir+'/model.599'
        limit_fee = 0.0006
        market_fee = 0.0016 + 0.002 # Market fee 0.0016 + slippage 0.2%
        order_type = 'market'
        signal_threshold = 1
        signal_scale = 100
        rsi_period = 14
        model_type = 'runNN1'
    elif conf == 'ETHUSDNN3':
        breakout = True
        sell_sl = True
#        buy_sl = True
        short = True
        leverage = 5
        min_equity = 0.02
        order_precision = 0
        exchange = 'KRAKEN'
#        datasource = 'kr'
#        kraken_pair = 'XETHZUSD'
        reload = True
        train = True
        train_pct = 0.75
        test_pct = 0.25
#        test_pct = 1
#        test_bars = 365
        units = 32
        epochs = 1000
#        model = cfgdir+'/model.nn'
        limit_fee = 0.0006
        market_fee = 0.0016 + 0.002 # Market fee 0.0016 + slippage 0.2%
        order_type = 'market'
        signal_threshold = 1
        signal_scale = 100
        rsi_period = 14
        model_type = 'runNN1'
        btc_data = True
#
# ***************************************** Active Models
# !!! Do not tune Active models - use new conf for tuning !!!
# !!! Scaler will be updated when tuning is run 
    elif conf == 'ETHUSDNN':
#        execute = True
        breakout = True
        sell_sl = True
        buy_sl = True
        short = True
        leverage = 5
        min_equity = 0.02
        order_precision = 0
        exchange = 'KRAKEN'
        datasource = 'kr'
        kraken_pair = 'XETHZUSD'
        reload = True
#        train = True
#        test_pct = 1
        test_bars = 365
        units = 32
        epochs = 20
        model = cfgdir+'/model.215'
        limit_fee = 0.0006
        market_fee = 0.0016 + 0.002 # Market fee 0.0016 + slippage 0.2%
        order_type = 'market'
    elif conf == 'ETHUSDNN1':
# model.603: 17 Nov SR: 5.81 Kraken: 187719 (epoch 189, batch size 1000) train 0.75, test 0.25, no buy SL
        breakout = True
        sell_sl = True
#        buy_sl = True
        short = True
        leverage = 5
        min_equity = 0.02
        order_precision = 0
        exchange = 'KRAKEN'
        datasource = 'kr'
        kraken_pair = 'XETHZUSD'
        reload = True
#        train = True
        train_pct = 0.75
#        test_pct = 0.25
#        test_pct = 1
        test_bars = 365
        units = 32
        epochs = 500
        model = cfgdir+'/model.603'
        limit_fee = 0.0006
        market_fee = 0.0016 + 0.002 # Market fee + slippage%
        order_type = 'market'
        signal_threshold = 1
        signal_scale = 100
        rsi_period = 14
        model_type = 'runNN1'

    if order_type == 'market':
        limit_fee = market_fee
    
    global file
    file = cfgdir+'/price.pkl'
    global q
    q = cfgdir+'/q.pkl'
    global tl
    tl = cfgdir+'/tl.pkl'
    print('')
    print('**************** Loaded Config for '+conf+' ****************')

