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

#np.random.seed(12345) # Set random seed so that results are reproducible
p_train = True
p_reload = True  # Reload price data or use existing
p_save_q = True
p_charts = True
p_stats = True
p_epochs = 50 # Number of iterations for training (best 50)
p_features = 4 # Number of features in state for Q table
p_feature_bins = 3 # Number of bins for feature (more bins tend to overfit)
p_train_pct = 1 # % of data used for training
p_test_pct = 1 # % of data used for testing
p_confidence = 1 # Best: 1
p_sma_period = 50
p_sma1_period = 20
p_adr_period = 20 # Average Daily Return period
p_hhll_period = 50 # Window for Highest High and Lowest Low (best: 20 - 50)
p_rsi_period = 14
# Number of actions (% of stock holdings) 2 for long only, 3 to add short
p_actions = 2
p_short = False
p_random_scale=0.00001 # Defines standard deviation for random Q values 
# α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
p_alpha = 0.2
# γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards. Best: 0.9
p_gamma = 0.9
# Probability to chose random action instead of best action from Q Table
# Best values: 0.2 - 0.5
p_epsilon = 0.5
p_start_balance = 1.0
p_currency = 'USD'

# Define Actions
actions = pd.DataFrame(np.linspace(-1 if p_short else 0, 1, p_actions))

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
#        qt = pd.read_csv(p_q, index_col=0)
    
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
    os.makedirs(os.path.dirname(p_file), exist_ok=True)
    pickle.dump(df, open(p_file, "wb" ))
#    df.to_csv(p_file)
    print(str(len(df))+' records loaded from '+p_exchange+' to '+p_file)

# Separate feature values to bins (numbers)
# Each bin has same number of feature values
def bin_feature(feature, test=False):
    binfile = 'data/'+p_conf+'/bin'+feature.name+'.pkl'
    if test:
        b = pickle.load(open(binfile, "rb" )) # Load bin config
        d = pd.cut(feature, bins=b, labels=False, include_lowest=True)
    else:
        d, b = pd.qcut(feature, p_feature_bins, duplicates='drop', labels=False, retbins=True)
        pickle.dump(b, open(binfile, "wb" )) # Save bin config
    return d

# Read Data from CSV and add features
def get_dataset(test=False, train_pct=p_train_pct, test_pct=p_test_pct):
    df = pickle.load(open(p_file, "rb" ))
#    df = pd.read_csv(p_file, parse_dates=True, usecols=['date', 'open', 'close', 'high', 'low'])

    # Add features to dataframe
    # Typical Features: close/sma, bollinger band, holding stock, return since entry
    df = df.assign(dr=df.close/df.close.shift(1)-1) # daily return
    df = df.assign(adr=ta.SMA(df, price='dr', timeperiod=p_adr_period))
#    df = df.assign(sma=ta.SMA(df, timeperiod=p_sma_period))
#    df = df.assign(sma1=ta.SMA(df, timeperiod=p_sma1_period))
#    df = df.assign(dsma=df.sma/df.sma.shift(1)-1)
#    df = df.assign(rsma=df.sma/df.sma1)
    df = df.assign(rsi=ta.RSI(df, timeperiod=p_rsi_period))
    df = df.assign(hh=df.high/ta.MAX(df, price='high', timeperiod=p_hhll_period))
    df = df.assign(ll=df.low/ta.MIN(df, price='low', timeperiod=p_hhll_period))
    df = df.dropna()
    # Map features to bins
    df = df.assign(binhh=bin_feature(df.hh, test))
    df = df.assign(binll=bin_feature(df.ll, test))
    df = df.assign(binadr=bin_feature(df.adr, test))
    df = df.assign(binrsi=bin_feature(df.rsi, test))
#    df = df.assign(bindsma=bin_feature(df.dsma))
#    df = df.assign(binrsma=bin_feature(df.rsma))

    # Separate Train / Test Datasets using train_pct number of rows
    if test:
        rows = int(len(df)*test_pct)
#        print("Get Test Dataset rows %s" % rows)
        return df.tail(rows).reset_index(drop=True)
    else:
        rows = int(len(df)*train_pct)
#        print("Get Train Dataset rows: %s" % rows)
        return df.head(rows).reset_index(drop=True)
    
# Calculate Discretised State based on features
def get_state(row):
    bins = p_feature_bins
    state = int(bins**3*row.binrsi+bins**2*row.binadr+bins*row.binhh+row.binll)
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
    
    if not test: qt = qt.assign(conf=bin_feature(qt.ratio))
    return df

def get_sr(df):
    return df.mean()/df.std()

def get_ret(df):
    return df.iloc[-1]/p_start_balance

def normalize(df):
    return df/df.at[0]

def show_result(df, title):
    # Thanks to: http://benalexkeen.com/bar-charts-in-matplotlib/
    df = df.assign(nclose=df.close/df.close.at[0]) # Normalise Price
    if p_charts:
        d = df.set_index('date')
        fig, ax = plt.subplots()
        ax.plot(d.nclose, label='Buy and Hold')
        ax.plot(d.total, label='QL')
#        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.title(title+' for '+p_conf)
        plt.ylabel('Return')
        plt.legend(loc='best')
        plt.show()
    print("QL R: %s" % (get_ret(df.total)))
    print("QL/BH SR: %s" % (get_sr(df.pnl)/get_sr(df.dr)))
    print("QL/BH R:  %s" % (get_ret(df.total)/get_ret(df.nclose)))
    print("Confidence: %s" % df.conf.mean())

def print_forecast(tdf):
    print()
    position = 'CASH' if tdf.cash.iloc[-1] > 0 else 'EQUITY'
    print('Current position: '+position)
    
    action = 'No action'
    if tdf.action.iloc[-1] != tdf.action.iloc[-2]:
        action = 'BUY' if tdf.action.iloc[-1] > 0 else 'SELL'
    print('Today action: '+action)

    state = tdf.state.iloc[-1]
    next_action, reward = get_action(state)
    conf = qt.conf.iloc[state]
    action = 'No action'
    if next_action != tdf.action.iloc[-1] and conf >= p_confidence:
        action = 'BUY' if next_action > 0 else 'SELL'
    print('Tomorrow estimate: '+action)

def train_model(df, tdf):
    global qt
    print("*** Training Model using "+p_ticker+" data ***") 
    print("Epochs: %s" % p_epochs) 

    max_r = 0
    for ii in range(p_epochs):
        # Train Model
        df = run_model(df)
        # Test Model   
        tdf = run_model(tdf, test=True)
        r = get_ret(tdf.total)
#        r = get_sr(tdf.pnl)
        if r > max_r:
            max_r = r
            max_q = qt.copy()
            print("Epoch: %s Max R: %s" % (ii, max_r))
    qt = max_q
    # Save Model
    if p_save_q:
        pickle.dump(qt, open('data/'+p_conf+'/q'+str(int(max_r))+'.pkl', "wb" ))

def run_forecast(conf):
    global tdf
    global df
    print()
    print('**************** Running forecast for '+conf+' ****************')
    load_config(conf)
    
    if p_reload: load_data() # Load Historical Price Data   
    tdf = get_dataset(test=True) # Read Test data
    if p_train: 
        df = get_dataset() # Read Train data
        train_model(df, tdf)
    
    tdf = run_model(tdf, test=True)
    if p_stats: show_result(tdf, "Test") # Test Result
    print_forecast(tdf) # Print Forecast

def load_config(conf):
    global p_ticker
    global p_q # File for Q table
    global p_file # File for price data 
    global p_spread # Spread percentage
    global qt
    global p_exchange
    global p_conf
    global p_train
    global p_reload
    global p_cfgdir
    global p_charts
    global p_stats
    
    p_conf = conf
    p_train = False
#    p_reload = False
#    p_charts = False
#    p_stats = False
    p_cfgdir = 'data/'+conf
    p_file = p_cfgdir+'/price.pkl'
    p_exchange = 'CCCAGG' # Average price from all exchanges
    p_q = p_cfgdir+'/q.pkl'

    if conf == 'AVGETHUSD': # 1616
        p_ticker = 'ETH'
        p_spread = 0.03 # eToro weekend spread
    if conf == 'BTFETHUSD': # 459
        p_exchange = 'Bitfinex'
        p_ticker = 'ETH'
        p_spread = 0
    elif conf == 'AVGBTCUSD': # 468734
        p_ticker = 'BTC'
        p_spread = 0.01 # eToro weekend spread
    elif conf == 'AVGXRPUSD': # 431
        p_ticker = 'XRP'
        p_spread = 0.04 # eToro weekend spread
    elif conf == 'AVGLTCUSD': # 3.67 
        p_ticker = 'LTC'
        p_spread = 0.04 # eToro weekend spread
    elif conf == 'TEST': 
        p_ticker = 'LTC'
        p_spread = 0.04 # eToro weekend spread

    qt = init_q() # Initialise Model

#run_forecast('BTFETHUSD')
#run_forecast('AVGETHUSD')
#run_forecast('AVGBTCUSD')
run_forecast('AVGLTCUSD')
#run_forecast('AVGXRPUSD')


# TODO:
# Install GIT / use GitHub

# Simple Trading System: Close/200MA, 200MA DR 

# Store bin config with Q

# Training: Load best Q and try to improve it. Save Q if improved

# Optimize loops. See https://www.datascience.com/blog/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/

# Add Ratio: Price/200MA. See: https://cointelegraph.com/news/ultra-rich-investor-trace-mayer-predicts-bitcoin-price-will-reach-27395-in-just-four-months

# Store execution history in csv
# Load best Q based on execution history

# Binary features allow more features to be used

# Add Volume to features

# Ensemble different strategies

# Solve Unknown State Problem: Find similar state

# Test model with train or test data?

# Reduce number of bins

# Fix short sell profit calculation. See: https://sixfigureinvesting.com/2014/03/short-selling-securities-selling-short/

# Implement Dyna Q

# Predict DR based on State (use R table)

# Implement Parameterised Feature List

# Automatic Data Reload (based on file date)

# Stop Iterating when Model Converges (define converge criteria)
# Converge Criteria: best result is not improved after n epochs (n is another parameter)
