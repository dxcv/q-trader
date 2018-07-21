# Inspired By: https://classroom.udacity.com/courses/ud501
# Install Brew: https://brew.sh/
# Install ta-lib: https://mrjbq7.github.io/ta-lib/install.html
# Ta Lib Doc: https://github.com/mrjbq7/ta-lib
# See Also: Implementation using keras-rl library
# https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/
# Sell/Buy orders are executed at last day close price
# Crypto Analysis: https://blog.patricktriest.com/analyzing-cryptocurrencies-python/

# pip install ccxt
#conda install theano
#conda install tensorflow
#conda install keras
#pip install -U numpy
 
import pandas as pd
import numpy as np
import time
import talib.abstract as ta
import talib
import matplotlib.pyplot as plt
import requests
import pickle
import os
import params as p
import exchange as ex
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Init Q table with small random values
def init_q():
    qt = pd.DataFrame()
    if p.train:
        qt = pd.DataFrame(np.random.normal(scale=p.random_scale, size=(p.feature_bins**p.features,p.actions)))
        qt['visits'] = 0
        qt['conf'] = 0
        qt['ratio'] = 0.0
    else:
        if os.path.isfile(p.q): qt = pickle.load(open(p.q, "rb" ))
    return qt


# Load Historical Price Data from Poloniex (currently not used)
def load_data_polo(): 
    period = '86400'  # 1 day candle
    #period = '14400'  # 4h candle
    start_date = "2010-01-01"
    end_date = "2100-01-01"
    dstart = str(int(time.mktime(time.strptime(start_date, "%Y-%m-%d"))))
    dend = str(int(time.mktime(time.strptime(end_date, "%Y-%m-%d"))))
    df = pd.read_json('https://poloniex.com/public?command=returnChartData&currencyPair='+p.ticker+'&start='+dstart+'&end='+dend+'&period='+period)
    df.to_csv(p.file)
#    print(str(len(df))+" records loaded from Poloniex for to "+p.file)

# Load Historical Price Data from Cryptocompare
# API Guide: https://medium.com/@agalea91/cryptocompare-api-quick-start-guide-ca4430a484d4
def load_data():
    if not p.reload: return pickle.load(open(p.file, "rb" ))
    
    if p.bar_period == 'day':
        period = 'histoday'
    elif p.bar_period == 'hour': 
        period = 'histohour'
    r = requests.get('https://min-api.cryptocompare.com/data/'+period
                     +'?fsym='+p.ticker+'&tsym='+p.currency
                     +'&allData=true&e='+p.exchange)
    df = pd.DataFrame(r.json()['Data'])
    df['date'] = pd.to_datetime(df['time'],unit='s')
    df.drop(['time', 'volumefrom', 'volumeto'], axis=1, inplace=True)
    os.makedirs(os.path.dirname(p.file), exist_ok=True)
    pickle.dump(df, open(p.file, "wb" ))
    print('Prices Loaded. Period:'+p.bar_period+' Rows:'+str(len(df))+' Date:'+str(df.date.iloc[-1]))
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
    
# Calculate Discretised State based on features
def get_state(row):
    bins = p.feature_bins
    if p.version == 1:
        state = int(bins**3*row.binrsi+bins**2*row.binadr+bins*row.binhh+row.binll)
    elif p.version == 2:
        state = int(bins**3*row.binrsma+bins**2*row.bindsma+bins*row.binrsi+row.binhhll)
    visits = qt.at[state, 'visits']
    conf = qt.at[state, 'conf']
    return state, visits, conf
    
# P Policy: P[s] = argmax(a)(Q[s,a])
def get_action(state, test=True):
    if (not test) and (np.random.random() < p.epsilon): 
        # choose random action with probability epsilon
        max_action = np.random.randint(0,p.actions)
    else: 
        #choose best action from Q(s,a) values
        max_action = int(qt.iloc[state,0:p.actions].idxmax(axis=1))
    
    max_reward = qt.iat[state, max_action]
    return max_action, max_reward

class Portfolio:
    def __init__(self, balance):
          self.cash = balance
          self.equity = 0.0
          self.short = 0.0
          self.total = balance    
    
    def upd_total(self):
        self.total = self.cash+self.equity+self.short


def buy_lot(pf, lot, short=False):
    if lot > pf.cash: lot = pf.cash
    pf.cash -= lot
    adj_lot = lot*(1-p.spread/2)
    if short: pf.short += adj_lot
    else: pf.equity += adj_lot
    
def sell_lot(pf, lot, short=False):
    if short:
        if lot > pf.short: lot = pf.short
        pf.short -= lot
    else:
        if lot > pf.equity: lot = pf.equity 
        pf.equity -= lot
    
    pf.cash = pf.cash + lot*(1-p.spread/2)

# Execute Action: buy or sell
def take_action(pf, action, dr):
    old_total = pf.total
    target = pf.total*actions.iat[action,0] # Target portfolio
    if target >= 0: # Long
        if pf.short > 0: sell_lot(pf, pf.short, True) # Close short positions first
        diff = target - pf.equity
        if diff > 0: buy_lot(pf, diff) 
        elif diff < 0: sell_lot(pf, -diff)
    else: # Short
        if pf.equity > 0: sell_lot(pf, pf.equity) # Close long positions first
        diff = -target - pf.short
        if diff > 0: buy_lot(pf, diff, True) 
        elif diff < 0: sell_lot(pf, -diff, True)

    # Calculate reward as a ratio to maximum daily return
    # reward = 1 - (1 + abs(dr))/(1 + dr*(equity-cash)/total)
        
    # Update Balance
    pf.equity = pf.equity*(1 + dr)
#    pf.short = pf.short*(1 - dr) This calculation is incorrect
    pf.upd_total()
    reward = pf.total/old_total - 1
    # Calculate Reward as pnl + cash dr
#    reward = (1+pnl)*(1-dr*pf.cash/old_total) - 1
    return reward        

# Update Rule Formula
# The formula for computing Q for any state-action pair <s, a>, given an experience tuple <s, a, s', r>, is:
# Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])])
#
# Here:
#
# r = R[s, a] is the immediate reward for taking action a in state s,
# γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards,
# s' is the resulting next state,
# argmaxa'(Q[s', a']) is the action that maximizes the Q-value among all possible actions a' from s', and,
# α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
#
def update_q(s, a, s1, r):
    action, reward = get_action(s1)
    q0 = qt.iloc[s, a]
    q1 = (1 - p.alpha)*q0 + p.alpha*(r + p.gamma*reward)
    qt.iloc[s, a] = q1
    qt.at[s1, 'visits'] += 1


# Iterate over data => Produce experience tuples: (s, a, s', r) => Update Q table
# In test mode do not update Q Table and no random actions (epsilon = 0)
def run_model(df, test=False):
    global qt
    df = df.assign(state=-1, visits=1, conf=0, action=0, equity=0.0, cash=0.0, total=0.0, pnl=0.0)
    pf = Portfolio(p.start_balance)
    
    for i, row in df.iterrows():
        if i == 0:            
            state, visits, conf = get_state(row) # Observe New State
            action = 0 # Do not take any action in first day
        else:
            old_state = state
            if test and conf == 0: # Use same action if confidence is low 
                action = action
            else:
                # Find Best Action based on previous state
                action, _ = get_action(old_state, test)
            # Take an Action and get Reward
            reward = take_action(pf, action, row.dr)
            # Observe New State
            state, visits, conf = get_state(row)
            # If training - update Q Table
            if not test: update_q(old_state, action, state, reward)
            df.at[i, 'pnl'] = reward
    
        df.at[i, 'visits'] = visits
        df.at[i, 'conf'] = conf
        df.at[i, 'action'] = action
        df.at[i, 'state'] = state
        df.at[i, 'equity'] = pf.equity
        df.at[i, 'cash'] = pf.cash
        df.at[i, 'total'] = pf.total
    
    if not test:
        qt['r'] = qt.visits * (qt.iloc[:,:p.actions].max(axis=1) - qt.iloc[:,:p.actions].min(axis=1))
        qt['ratio'] = qt.r / qt.r.sum()
        qt['conf'] = (qt['ratio'] > p.ratio).astype('int')
             
    return df

# Sharpe Ratio Calculation
# See also: https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement
def get_sr(df):
    return df.mean()/(df.std()+0.000000000000001) # Add small number to avoid division by 0

def get_ret(df):
    return df.iloc[-1]/df.iloc[0]

def normalize(df):
    return df/df.at[0]

def train_model(df, tdf):
    global qt
    print("*** Training Model using "+p.ticker+" data. Epochs: %s ***" % p.epochs) 

    max_r = 0
    max_q = qt
    for ii in range(p.epochs):
        # Train Model
        df = run_model(df)
        # Test Model   
        tdf = run_model(tdf, test=True)
        if p.train_goal == 'R':
            r = get_ret(tdf.total)
        else:
            r = get_sr(tdf.pnl)
#        print("Epoch: %s %s: %s" % (ii, p.train_goal, r))
        if r > max_r:
            max_r = r
            max_q = qt.copy()
            print("*** Epoch: %s Max %s: %s" % (ii, p.train_goal, max_r))
    
    qt = max_q
    if max_r > p.max_r:
        print("*** New Best Model Found! Best R: %s" % (max_r))
        # Save Model
        pickle.dump(qt, open(p.cfgdir+'/q'+str(int(1000*max_r))+'.pkl', "wb" ))

def show_result(df, title):
    # Thanks to: http://benalexkeen.com/bar-charts-in-matplotlib/
    if p.result_size > 0: df = df.tail(p.result_size).reset_index(drop=True)
    df['nclose'] = normalize(df.close) # Normalise Price
    df['ntotal'] = normalize(df.total) # Normalise Price
    if p.charts:
        d = df.set_index('date')
        d['signal'] = d.action-d.action.shift(1)        
        fig, ax = plt.subplots()
        ax.plot(d.nclose, label='Buy and Hold')
        ax.plot(d.ntotal, label='QL', color='red')
        
        # Plot buy signals
        ax.plot(d.loc[d.signal == 1].index, d.ntotal[d.signal == 1], '^', 
                markersize=10, color='m', label='BUY')
        # Plot sell signals
        ax.plot(d.loc[d.signal == -1].index, d.ntotal[d.signal == -1], 'v', 
                markersize=10, color='k', label='SELL')
        
        fig.autofmt_xdate()
        plt.title(title+' for '+p.conf)
        plt.ylabel('Return')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    qlr = get_ret(df.ntotal)
    qlsr = get_sr(df.pnl)
    bhr = get_ret(df.nclose)
    bhsr = get_sr(df.dr)
    print("R: %.2f SR: %.3f QL/BH R: %.2f QL/BH SR: %.2f" % (qlr, qlsr, qlr/bhr, qlsr/bhsr))
    print("AVG Confidence: %.2f" % df.conf.mean())
    print('QT States: %s Valid: %s Confident: %s' % 
          (len(qt), len(qt[qt.visits > 0]), len(qt[qt.conf >= 1])))

def get_today_action(tdf):
    action = 'HOLD'
    if tdf.action.iloc[-1] != tdf.action.iloc[-2]:
        action = 'BUY' if tdf.action.iloc[-1] > 0 else 'SELL'
    return action

def print_forecast(tdf):
    print()
    position = p.currency if tdf.cash.iloc[-1] > 0 else p.ticker
    print('Current position: '+position)
    print('Today: '+get_today_action(tdf))

    state = tdf.state.iloc[-1]
    next_action, reward = get_action(state)
    conf = qt.conf.iloc[state]
    action = 'HOLD'
    if next_action != tdf.action.iloc[-1] and conf >= 1:
        action = 'BUY' if next_action > 0 else 'SELL'
    print('Tomorrow: '+action)

class TradeLog:
    def __init__(self):
        self.cash = p.start_balance
        self.equity = 0.0
        columns = ['date', 'action', 'cash', 'equity', 'price', 'cash_bal', 'equity_bal']
        self.log = pd.DataFrame(columns=columns)
    
    def log_trade(self, action, cash, equity):
        price = abs(cash)/abs(equity)
        self.cash += cash
        self.equity += equity
        row = [{'date': dt.datetime.now(),'action':action, 
            'cash':cash, 'equity':equity, 'price':price, 
            'cash_bal':self.cash, 'equity_bal':self.equity}]
        self.log = self.log.append(row, ignore_index=True)

def execute_action():
    print('!!!EXECUTE MODE!!!')
    action = get_today_action(tdf)
    if action == 'HOLD': return
    amount = tl.cash if action == 'buy' else tl.equity
    cash, equity = ex.market_order(action, amount)
    tl.log_trade(action, cash, equity) # Update trade log
    pickle.dump(tl, open(p.tl, "wb" ))

def run_forecast(conf, seed = None):
    global tdf
    global df

    if seed is not None: np.random.seed(seed)
    print('')
    print('**************** Running model for '+conf+' ****************')
    load_config(conf)
    
    load_data() # Load Historical Price Data   
    # This needs to run before test dataset as it generates bin config
    if p.train: df = get_dataset() # Read Train data. 
    tdf = get_dataset(test=True) # Read Test data
    if p.train: train_model(df, tdf)
    
    tdf = run_model(tdf, test=True)
    if p.stats: show_result(tdf, "Test") # Test Result
    print_forecast(tdf) # Print Forecast
    if p.execute: execute_action()

def load_config(conf):
    global actions
    global tl
    global qt # Q Table
    
#    np.random.seed(12345) # Set random seed so that results are reproducible
    p.random_scale = 0.00001 # Defines standard deviation for random Q values 
    p.start_balance = 1.0
    p.short = False # Short calculation is currently incorrect hense disabled
    p.actions = 2 # Number of actions (% of stock holdings) 2 for long only, 3 to add short
    # α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
    p.alpha = 0.2
    # γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards. Best: 0.9
    p.gamma = 0.9
    # Probability to chose random action instead of best action from Q Table. Best values: 0.2 - 0.5
    p.epsilon = 0.5
    p.train = False # Train model
    p.reload = True # Reload price data or use existing  
    p.charts = False # Plot charts
    p.stats = True # Show model stats
    p.epochs = 100 # Number of iterations for training (best 50)
    p.features = 4 # Number of features in state for Q table
    p.feature_bins = 3 # Number of bins for feature (more bins tend to overfit)
    p.train_pct = 1 # % of data used for training
    p.test_pct = 1 # % of data used for testing
    
    p.max_r = 0
    p.conf = conf
    p.ticker = p.conf[0:3]
    p.currency = p.conf[3:6]
    p.cfgdir = 'data/'+p.conf
    p.version = 2 # Model version
    p.sma_period = 50 # Best: 50
    p.adr_period = 20 # Average Daily Return period
    p.hh_period = 50 # Window for Highest High (best: 20 - 50)
    p.ll_period = 50 # Window for Lowest Low (best: 20 - 50)
    p.rsi_period = 50
    p.exchange = 'CCCAGG' # Average price from all exchanges
    p.execute = False
    p.order_size = 0 # Maximum order size in equity
    p.result_size = 0
    p.order_wait = 10
    p.min_cash = 1
    p.min_equity = 0.001
    p.bar_period = 'day' # Price bar period: day or hour
    p.max_bars = 1000 # Number of bars to use for training
    p.train_goal = 'R' # Maximize Return
    p.spread = 0.004 # Bitfinex fee
    p.ratio = 0 # Min ratio for Q table to take an action

    if conf == 'BTCUSD': # R: 180.23 SR: 0.180 QL/BH R: 6.79 QL/BH SR: 1.80
        p.max_r = 180
        p.version = 1
    elif conf == 'ETHUSD': # R: 6984.42 SR: 0.164 QL/BH R: 8.94 QL/BH SR: 1.30
        p.max_r = 6984
    elif conf == 'ETHBTC': # R: 1020.86 SR: 0.148 QL/BH R: 36.71 QL/BH SR: 1.81
        p.version = 1
        p.max_r = 1020
    elif conf == 'ETHUSDTEST': #
#        p.train = True
        p.max_r = 1.7
#        p.reload = False
        p.max_bars = 0
#        p.train_pct = 0.8
#        p.test_pct = 0.2
#        p.charts = True
    elif conf == 'ETHUSDNN':
        p.max_r = 18228
#        p.train = True
        p.max_bars = 0
        p.train_pct = 0.8
        p.test_pct = 0.2
#        p.shuffle = True
#        p.reload = False
        p.epochs = 500
        p.model = p.cfgdir+'/best.nn'
        
    if p.train:
        p.charts = True
        p.stats = True
        
    p.file = p.cfgdir+'/price.pkl'
    p.q = p.cfgdir+'/q.pkl'
    p.tl = p.cfgdir+'/tl.pkl'

    qt = init_q() # Initialise Model
    actions = pd.DataFrame(np.linspace(-1 if p.short else 0, 1, p.actions))
    if os.path.isfile(p.tl):
        tl = pickle.load(open(p.tl, "rb" ))
    else:
        tl = TradeLog()

def run_batch(conf, instances = 1):
    if instances == 1:
        run_forecast(conf)
        return
    ts = time.time()
    run_forecast_a = partial(run_forecast, conf) # Returning a function of a single argument
    with ProcessPoolExecutor() as executor: # Run multiple processes
        executor.map(run_forecast_a, range(instances))
         
    print('Took %s', time.time() - ts)

# Source:
# https://www.quantinsti.com/blog/artificial-neural-network-python-using-keras-predicting-stock-price-movement/
def runNN(conf):
    global trade_dataset
    
    load_config(conf)
    dataset = load_data()
    
    # Calculate Features
    # Tomorrow Return - this should not be included in training set
    dataset['TR'] = (dataset['close']/dataset['close'].shift(1)).shift(-1)
    dataset['H-L'] = dataset['high'] - dataset['low']
    dataset['O-C'] = dataset['close'] - dataset['open']
    dataset['3day MA'] = dataset['close'].shift(1).rolling(window = 3).mean()
    dataset['10day MA'] = dataset['close'].shift(1).rolling(window = 10).mean()
    dataset['30day MA'] = dataset['close'].shift(1).rolling(window = 30).mean()
    dataset['Std_dev']= dataset['close'].rolling(5).std()
    dataset['RSI'] = talib.RSI(dataset['close'].values, timeperiod = 9)
    dataset['Williams %R'] = talib.WILLR(dataset['high'].values, dataset['low'].values, dataset['close'].values, 7)
    
    # Predicted value is whether price will rise
    dataset['Price_Rise'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0)
    dataset = dataset.dropna()
    
    # Shuffle rows in dataset
    if p.shuffle: dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    # Separate input from output
    X = dataset.iloc[:, -9:-1]
    y = dataset.iloc[:, -1]
    
    # Separate train from test
    train_split = int(len(dataset)*p.train_pct)
    test_split = int(len(dataset)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Building Neural Network
    from keras.models import Sequential
    from keras.layers import Dense
#    from keras.layers import Dropout
    #from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    
    # Early stopping  
    #es = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='max')
    model = p.cfgdir+'/model.nn'
    cp = ModelCheckpoint(model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
     
    classifier = Sequential()
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
#    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    if p.train:
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        history = classifier.fit(X_train, y_train, batch_size = 10, epochs = p.epochs, callbacks=[cp], validation_data=(X_test, y_test))
    
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
    
    # Compile model (required to make predictions)
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # Predicting The Price
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    dataset['y_pred'] = np.NaN
    dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
    trade_dataset = dataset.dropna().copy()
    
    # If price is predicted to drop - sell (no short selling)
    trade_dataset['SR'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['TR'], 1)
    trade_dataset['CMR'] = np.cumprod(trade_dataset['TR'])
    trade_dataset['CSR'] = np.cumprod(trade_dataset['SR'])
    
    # Plot the graph
#    trade_dataset = trade_dataset.set_index('date')
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    plt.plot(trade_dataset['CMR'], color='r', label='Market Returns')
    plt.plot(trade_dataset['CSR'], color='g', label='Strategy Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print('Trade Frequency: '+ str(len(trade_dataset[trade_dataset['y_pred'] != trade_dataset['y_pred'].shift(-1)])/len(trade_dataset)))

def run():
    #q.run_batch('BTCUSD') # Bitcoin: Stop trading low profit strategy?
    #q.run_batch('ETHUSD') # Ethereum
    #q.run_batch('ETHBTC') # Better than ETHUSD if counted in USD terms
    
#    run_batch('ETHUSDTEST')    
    runNN('ETHUSDNN')

run()


# TIP: Sell permanently when state 80 is changed to other state

# TODO:
# Separate train_model and run_model procedures

# Calculate R in USD

# Trade with daily averege price: split order in small chunks and execute during day

# Integrate with Telegram Bot

# Populate Trade Log for train/test mode

# Use hourly data for daily strategy. Any hour can be used for day end
# Build 24 daily strategies (one for each hour) Ensemble strategy for each hour

# Use Monte Carlo to find best parameters 

# Ensemble strategy: avg of best Q tables

# Add month/day to state

# Test price change scenario

# Sentiment analysis: https://github.com/Crypto-AI/Stocktalk

# Training: Load best Q and try to improve it. Save Q if improved

# Optimize loops. See https://www.datascience.com/blog/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/

# Store execution history in csv
# Load best Q based on execution history

# Add Volume to features

# Solve Unknown State Problem: Find similar state

# Test model with train or test data?

# Fix short sell profit calculation. See: https://sixfigureinvesting.com/2014/03/short-selling-securities-selling-short/

# Implement Dyna Q

# Predict DR based on State (use R table)

# Implement Parameterised Feature List
# Use function list: https://realpython.com/blog/python/primer-on-python-decorators/
# Lambda, map, reduce: https://www.python-course.eu/lambda.php

# Automatic Data Reload (based on file date)

# Stop Iterating when Model Converges (define converge criteria)
# Converge Criteria: best result is not improved after n epochs (n is another parameter)
