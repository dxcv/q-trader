# Inspired By: https://classroom.udacity.com/courses/ud501
# Install Brew: https://brew.sh/
# Install ta-lib: https://mrjbq7.github.io/ta-lib/install.html
# Ta Lib Doc: https://github.com/mrjbq7/ta-lib
# See Also: Implementation using keras-rl library
# https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/
# Sell/Buy orders are executed at last day close price
 
import pandas as pd
import numpy as np
import time
import talib.abstract as ta
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import requests
import pickle
import os

# Init Q table with small random values
def init_q():
    if p_train:
        qt = pd.DataFrame(np.random.normal(scale=p_random_scale, size=(p_feature_bins**p_features,p_actions)))
        qt = qt.assign(visits=0)
        qt = qt.assign(conf=0)
        qt = qt.assign(ratio=0.0)
    else: 
        print("Loading Q from "+p_q)
        qt = pickle.load(open(p_q, "rb" ))
    
    return qt

# Load Historical Price Data from Poloniex (currently not used)
def load_data_polo(): 
    period = '86400'  # 1 day candle
    #period = '14400'  # 4h candle
    start_date = "2010-01-01"
    end_date = "2100-01-01"
    dstart = str(int(time.mktime(time.strptime(start_date, "%Y-%m-%d"))))
    dend = str(int(time.mktime(time.strptime(end_date, "%Y-%m-%d"))))
    df = pd.read_json('https://poloniex.com/public?command=returnChartData&currencyPair='+p_ticker+'&start='+dstart+'&end='+dend+'&period='+period)
    df.to_csv(p_file)
    print(str(len(df))+" records loaded from Poloniex for to "+p_file)

# Load Historical Price Data from Cryptocompare
def load_data():
#    period = 'histohour'
    period = 'histoday'
    r = requests.get('https://min-api.cryptocompare.com/data/'+period+'?fsym='+p_ticker+'&tsym='+p_currency+'&allData=true&e='+p_exchange)
    df = pd.DataFrame(r.json()['Data'])
    df = df.assign(date=pd.to_datetime(df['time'],unit='s'))
    df.drop(['time', 'volumefrom', 'volumeto'], axis=1, inplace=True)
    os.makedirs(os.path.dirname(p_file), exist_ok=True)
    pickle.dump(df, open(p_file, "wb" ))
    print(str(len(df))+' records loaded from '+p_exchange+' to '+p_file)

# Separate feature values to bins (numbers)
# Each bin has same number of feature values
def bin_feature(feature, test=False, bins=p_feature_bins):
    binfile = 'data/'+p_conf+'/bin'+feature.name+'.pkl'
    if test:
        b = pickle.load(open(binfile, "rb" )) # Load bin config
        d = pd.cut(feature, bins=b, labels=False, include_lowest=True)
    else:
        d, b = pd.qcut(feature, bins, duplicates='drop', labels=False, retbins=True)
        pickle.dump(b, open(binfile, "wb" )) # Save bin config
    return d

# Read Price Data and add features
def get_dataset(test=False, train_pct=p_train_pct, test_pct=p_test_pct):
    df = pickle.load(open(p_file, "rb" ))
    
#    df.iloc[-1, df.columns.get_loc('close')] = test_price

    # Add features to dataframe
    # Typical Features: close/sma, bollinger band, holding stock, return since entry
    df = df.assign(dr=df.close/df.close.shift(1)-1) # daily return
    df = df.assign(adr=ta.SMA(df, price='dr', timeperiod=p_adr_period))
    df = df.assign(sma=ta.SMA(df, price='close', timeperiod=p_sma_period))
    df = df.assign(dsma=df.sma/df.sma.shift(1)-1)
    df = df.assign(rsma=df.close/df.sma)
    df = df.assign(rsi=ta.RSI(df, price='close', timeperiod=p_rsi_period))
    df = df.assign(hh=df.high/ta.MAX(df, price='high', timeperiod=p_hhll_period))
    df = df.assign(ll=df.low/ta.MIN(df, price='low', timeperiod=p_hhll_period))
    df = df.assign(hhll=(df.high+df.low)/(df.high/df.hh+df.low/df.ll))
    df = df.dropna()
    # Map features to bins
    df = df.assign(binrsi=bin_feature(df.rsi, test))
    if p_version == 1:
        df = df.assign(binadr=bin_feature(df.adr, test))
        df = df.assign(binhh=bin_feature(df.hh, test))
        df = df.assign(binll=bin_feature(df.ll, test))
    elif p_version == 2:
        df = df.assign(bindsma=bin_feature(df.dsma, test))
        df = df.assign(binrsma=bin_feature(df.rsma, test))
        df = df.assign(binhhll=bin_feature(df.hhll, test))

    # Separate Train / Test Datasets using train_pct number of rows
    if test:
        rows = int(len(df)*test_pct)
        return df.tail(rows).reset_index(drop=True)
    else:
        rows = int(len(df)*train_pct)
        return df.head(rows).reset_index(drop=True)
    
# Calculate Discretised State based on features
def get_state(row):
    bins = p_feature_bins
    if p_version == 1:
        state = int(bins**3*row.binrsi+bins**2*row.binadr+bins*row.binhh+row.binll)
    elif p_version == 2:
        state = int(bins**3*row.binrsma+bins**2*row.bindsma+bins*row.binrsi+row.binhhll)
    visits = qt.at[state, 'visits']
    conf = qt.at[state, 'conf']
    return state, visits, conf
    
# P Policy: P[s] = argmax(a)(Q[s,a])
def get_action(state, test=True):
    if (not test) and (np.random.random() < p_epsilon): 
        # choose random action with probability epsilon
        max_action = np.random.randint(0,p_actions)
    else: 
        #choose best action from Q(s,a) values
        max_action = int(qt.iloc[state,0:p_actions].idxmax(axis=1))
    
    max_reward = qt.iloc[state, max_action]
    return max_action, max_reward

class Portfolio:
    cash = p_start_balance
    equity = 0.0
    short = 0.0
    total = cash
    
    def upd_total(self):
        self.total = self.cash+self.equity+self.short


def buy_lot(pf, lot, short=False):
    if lot > pf.cash: lot = pf.cash
    pf.cash -= lot
    adj_lot = lot*(1-p_spread/2)
    if short: pf.short += adj_lot
    else: pf.equity += adj_lot
    
def sell_lot(pf, lot, short=False):
    if short:
        if lot > pf.short: lot = pf.short
        pf.short -= lot
    else:
        if lot > pf.equity: lot = pf.equity 
        pf.equity -= lot
    
    pf.cash = pf.cash + lot*(1-p_spread/2)

# Execute Action: buy or sell
def take_action(pf, action, dr):
    old_total = pf.total
    target = pf.total*actions.iloc[action,0] # Target portfolio
    if target >= 0: # Long
        if pf.short > 0: sell_lot(pf, pf.short, True) # Close short positions first
        diff = target - pf.equity
        if diff >= 0: buy_lot(pf, diff) 
        else: sell_lot(pf, -diff)
    else: # Short
        if pf.equity > 0: sell_lot(pf, pf.equity) # Close long positions first
        diff = -target - pf.short
        if diff >= 0: buy_lot(pf, diff, True) 
        else: sell_lot(pf, -diff, True)

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
def update_q(s, a, s1, r, alpha=p_alpha, gamma=p_gamma):
    action, reward = get_action(s1)
    q0 = qt.iloc[s, a]
    q1 = (1 - alpha)*q0 + alpha*(r + gamma*reward)
    qt.iloc[s, a] = q1
    qt.at[s1, 'visits'] += 1

    max_action = int(qt.iloc[s,0:p_actions].idxmax(axis=1))
    min_action = int(qt.iloc[s,0:p_actions].idxmin(axis=1))
    max_reward = qt.iloc[s, max_action]
    min_reward = qt.iloc[s, min_action]
    ratio = max_reward - min_reward
    qt.at[s, 'ratio'] = ratio

    return ratio

# Iterate over data => Produce experience tuples: (s, a, s', r) => Update Q table
# In test mode do not update Q Table and no random actions (epsilon = 0)
def run_model(df, test=False):
    global qt
    df = df.assign(state=-1, visits=1, conf=0, action=0, equity=0.0, cash=0.0, total=0.0, pnl=0.0)
    pf = Portfolio()
    
    for i, row in df.iterrows():
        if i == 0:            
            state, visits, conf = get_state(row) # Observe New State
            action = 0 # Do not take any action in first day
        else:
            old_state = state
            if not (test and conf < p_confidence): # Use same action if confidence is low 
                # Find Best Action based on previous state
                action, expr = get_action(old_state, test)    
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
    
    if not test: qt = qt.assign(conf=bin_feature(qt.ratio, bins=3))
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
    print("*** Training Model using "+p_ticker+" data. Epochs: %s ***" % p_epochs) 

    max_r = 0
    max_q = qt
    for ii in range(p_epochs):
        # Train Model
        df = run_model(df)
        # Test Model   
        tdf = run_model(tdf, test=True)
#        r = get_ret(tdf.total)
        r = get_sr(tdf.pnl)
        if r > max_r:
            max_r = r
            max_q = qt.copy()
            print("Epoch: %s Max SR: %s" % (ii, max_r))
    
    qt = max_q
    if max_r > p_max_r:
        print("*** New Best Model Found! Best SR: %s" % (max_r))
        # Save Model
        pickle.dump(qt, open('data/'+p_conf+'/q'+str(int(1000*max_r))+'.pkl', "wb" ))

def show_result(df, title):
    # Thanks to: http://benalexkeen.com/bar-charts-in-matplotlib/
    df = df.assign(nclose=normalize(df.close)) # Normalise Price
    if p_charts:
        d = df.set_index('date')
        fig, ax = plt.subplots()
        ax.plot(d.nclose, label='Buy and Hold')
        ax.plot(d.total, label='QL', color='red')
#        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.title(title+' for '+p_conf)
        plt.ylabel('Return')
        plt.legend(loc='best')
        plt.show()
    
    qlr = get_ret(df.total)
    qlsr = get_sr(df.pnl)
    bhr = get_ret(df.nclose)
    bhsr = get_sr(df.dr)
    print("R: %.2f SR: %.3f QL/BH R: %.2f QL/BH SR: %.2f" % (qlr, qlsr, qlr/bhr, qlsr/bhsr))
    print("AVG Confidence: %.2f" % df.conf.mean())
    print('QT States: %s Valid: %s Confident: %s' % 
          (len(qt), len(qt[qt.visits > 0]), len(qt[qt.conf >= p_confidence])))

def print_forecast(tdf):
    print()
    position = p_currency if tdf.cash.iloc[-1] > 0 else p_ticker
    print('Current position: '+position)
    
    action = 'No action'
    if tdf.action.iloc[-1] != tdf.action.iloc[-2]:
        action = 'BUY' if tdf.action.iloc[-1] > 0 else 'SELL'
    print('Today: '+action)

    state = tdf.state.iloc[-1]
    next_action, reward = get_action(state)
    conf = qt.conf.iloc[state]
    action = 'No action'
    if next_action != tdf.action.iloc[-1] and conf >= p_confidence:
        action = 'BUY' if next_action > 0 else 'SELL'
    print('Tomorrow: '+action)


def run_forecast(conf):
    global tdf
    global df
    print()
    print('**************** Running forecast for '+conf+' ****************')
    load_config(conf)
    
    if p_reload: load_data() # Load Historical Price Data   
    # This needs to run before test dataset as it generates bin config
    if p_train: df = get_dataset() # Read Train data. 
    tdf = get_dataset(test=True) # Read Test data
    if p_train: train_model(df, tdf)
    
    tdf = run_model(tdf, test=True)
    if p_stats: show_result(tdf, "Test") # Test Result
    print_forecast(tdf) # Print Forecast

def predict_dr(conf):
    tdf = get_dataset(test=True)
    tdf = tdf.assign(nextdr=tdf.dr.shift(-1))
    tdf = tdf.dropna()

#    x = tdf[['dr', 'rsi', 'dsma', 'rsma', 'hhll']].values
    x = tdf[['rsi']].values
    y = tdf['nextdr'].values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    
    # Building the optimal model using Backward Elimination
    import statsmodels.formula.api as sm
    x = np.append(arr = np.ones((len(x), 1)).astype(int), values=x, axis=1)
    x_opt = x[:, [2]]
    regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
    regressor_OLS.summary()

    # Predicting the Test set results
    tdf = tdf.assign(preddr=regressor.predict(x))
    tdf = tdf.assign(preddelta=tdf.preddr-tdf.nextdr)
    return tdf

def load_config(conf):
    global p_ticker
    global p_q # File for Q table
    global p_file # File for price data 
    global p_spread # Spread percentage
    global p_exchange
    global p_conf
    global p_train # Train model
    global p_reload # Reload price data or use existing
    global p_cfgdir
    global p_charts # Plot charts
    global p_stats # Show model stats
    global p_version
    global p_max_r
    global p_confidence # Best: 1
    global p_epochs # Number of iterations for training (best 50)
    global p_features # Number of features in state for Q table
    global p_feature_bins # Number of bins for feature (more bins tend to overfit)
    global p_actions  # Number of actions (% of stock holdings) 2 for long only, 3 to add short
    global p_train_pct # % of data used for training
    global p_test_pct # % of data used for testing
    global p_sma_period # Best: 50
    global p_adr_period # Average Daily Return period
    global p_hhll_period # Window for Highest High and Lowest Low (best: 20 - 50)
    global p_rsi_period
    global p_short
    global p_random_scale # Defines standard deviation for random Q values
    global p_alpha # α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
    global p_gamma # γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards. Best: 0.9
    global p_epsilon # Probability to chose random action instead of best action from Q Table. Best values: 0.2 - 0.5
    global p_start_balance
    global p_currency
    global p_max_r # Best result achieved for Q model
    global qt
    global actions

    #np.random.seed(12345) # Set random seed so that results are reproducible
    p_random_scale=0.00001  
    p_start_balance = 1.0
    p_currency = 'USD'
    p_max_r = 0
    p_short = False # Short calculation is currently incorrect hense disabled
    p_actions = 2
    p_alpha = 0.2
    p_gamma = 0.9
    p_epsilon = 0.5
    p_train = False 
    p_reload = False   
    p_charts = False
    p_stats = False 
    p_epochs = 50 
    p_features = 4 
    p_feature_bins = 3 
    p_version = 2 
    p_train_pct = 1 
    p_test_pct = 1 
    p_confidence = 1 
    p_sma_period = 50 
    p_adr_period = 20 
    p_hhll_period = 50 
    p_rsi_period = 14
    
    p_conf = conf
    p_cfgdir = 'data/'+conf
    p_file = p_cfgdir+'/price.pkl'
    p_exchange = 'CCCAGG' # Average price from all exchanges
    p_q = p_cfgdir+'/q.pkl'
#    p_train = True
    p_reload = True
#    p_charts = True
#    p_stats = True

    if conf == 'AVGETHUSD': 
        p_max_r = 0.184 # 2281
        p_ticker = 'ETH'
        p_spread = 0.03 # eToro weekend spread
    if conf == 'BTFETHUSD': 
        p_version = 1
        p_max_r = 0.228 # 465
        p_exchange = 'Bitfinex'
        p_ticker = 'ETH'
        p_spread = 0
    elif conf == 'AVGBTCUSD': 
        p_max_r = 0.139 # 1761898
        p_ticker = 'BTC'
        p_spread = 0.01 # eToro weekend spread
    elif conf == 'AVGXRPUSD': 
        p_confidence = 2
        p_max_r = 0.107 # 1379
        p_ticker = 'XRP'
        p_spread = 0.04 # eToro weekend spread
        p_rsi_period = 50
    elif conf == 'AVGLTCUSD': 
        p_confidence = 2
        p_max_r = 0.067 # 20
        p_ticker = 'LTC'
        p_spread = 0.04 # eToro weekend spread
        p_rsi_period = 50
    elif conf == 'AVGETHBTC':
        p_version = 1
        p_currency = 'BTC'
        p_ticker = 'ETH'
        p_max_r = 0.164 # 716
        p_spread = 0.01 # Exodus spread
        p_rsi_period = 50
    elif conf == 'TEST':
        p_version = 1
        p_currency = 'BTC'
        p_ticker = 'ETH'
        p_max_r = 0.158 # 541
        p_spread = 0.01 # Exodus spread
        p_rsi_period = 50
        p_epochs = 100 

    actions = pd.DataFrame(np.linspace(-1 if p_short else 0, 1, p_actions))
    qt = init_q() # Initialise Model

run_forecast('BTFETHUSD')
run_forecast('AVGETHUSD')
run_forecast('AVGBTCUSD')
run_forecast('AVGXRPUSD')
run_forecast('AVGETHBTC')

#run_forecast('AVGLTCUSD')
#run_forecast('TEST')


# TODO:
# !!! Test Partial equity buy 
# Add new action: NA (no action)/ test with confidence = 0 

# Store bin config with Q

# Test price change scenario

# Training: Load best Q and try to improve it. Save Q if improved

# Optimize loops. See https://www.datascience.com/blog/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/

# Store execution history in csv
# Load best Q based on execution history

# Add Volume to features

# Ensemble different strategies

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
