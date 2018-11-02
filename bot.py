#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:47:41 2018

@author: igor
"""

import ccxt
import time
import qlib as q
import params as p
import secrets as s
import tele as t

sleep_time = 60*60 # Sleep for 1 hour

ex = ccxt.kraken({
    'apiKey': s.exchange_api_key,
    'secret': s.exchange_sk
})

old_bar = None

def print(msg):
    t.send_msg(str(msg))

def execute():
    global old_bar
    pair = p.ticker + '/' + p.currency
    # Get latest price data
    df = q.load_data()
    new_bar = df.iloc[-2]
    if old_bar is None:
        print('Just started. Waiting for new bar')
    elif new_bar.equals(old_bar):        
        pass # Old bar - no action
    elif new_bar.date == old_bar.date:
        print('Warning!!!! Old bar has changed')
        print('Old Bar:')
        print(old_bar)
        print('New Bar:')
        print(new_bar)
    else:
        print('New bar found:')
        print(new_bar)
    old_bar = new_bar

    ticker = ex.fetch_ticker(pair)
    print('Current Price: ' + str(ticker['last']))

def run_bot(conf):
    q.init(conf)
    while True:
        execute()
        time.sleep(sleep_time) 

try:
    run_bot('ETHUSDNN')
finally:
    print('I am finished')
    t.cleanup()

# Fetch Balance
# balance = ex.fetch_balance()

# Normal Buy 0.1 ETH at market rate (no leverage)
# order = ex.create_market_buy_order('ETH/USD', 0.1)

# Normal Sell 0.1 ETH at market rate (no leverage)
# order = ex.create_market_sell_order('ETH/USD', 0.1)

# Leverage Buy
# order = ex.create_market_buy_order('ETH/USD', 0.01, {'leverage': 2})

# Leverage Sell
# order = ex.create_market_sell_order('ETH/USD', 0.01, {'leverage': 2})
