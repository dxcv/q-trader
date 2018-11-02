#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:47:41 2018

@author: igor
"""

import time
import qlib as q
import tele as t
import exchange as x

sleep_time = 60*15 # Sleep for 15 min

old_bar = None

def print(msg):
    t.send_msg(str(msg))

def execute(conf):
    global old_bar
    # Execute model
    q.runNN(conf)
    new_bar = q.td.iloc[-1]
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
    
    print('Current Price: ' + str(x.get_price()))

def run_bot(conf):
    while True:
        execute(conf)
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
