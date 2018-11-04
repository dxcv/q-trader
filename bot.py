#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:47:41 2018

@author: igor
"""

import qlib as q
import tele as t
import exchange as x
import datetime as dt
import time
import params as p

def send(msg):
    print(msg)
    t.send_msg(str(msg))

def get_signal(conf):
    no_signal = True

    while no_signal:
        q.runNN(conf)
        signal = q.get_signal()
        if dt.datetime.today() > signal['end']:
            send('Signal has expired. Waiting for new one ...')
            time.sleep(p.sleep_interval)
        else:
            no_signal = False

    return signal

def execute(conf):
    send('I am started')
    signal = get_signal(conf)
    print(str(signal))
 
    if signal['hold']:
        send('Hold - no action is required')
    elif signal['signal'] == 'Buy':
        send('Received Buy Signal')
        send('Closing Margin Sell Position')
        res = x.market_order('Buy', True)
        send('Balance: ' + str(res))

        send('Opening Buy Position')
        res = x.market_order('Buy')
        send('Balance: ' + str(res))
    elif signal['signal'] == 'Sell':
        send('Received Sell Signal')
        send('Closing Buy Position')
        res = x.market_order('Sell')
        send('Balance: ' + str(res))

        send('Opening Margin Sell Position')
        res = x.market_order('Sell', True)
        send('Balance: ' + str(res))

try:
    execute('ETHUSDNN')
except Exception as e:
    send('An error occured')
    print(e)
finally:
    send('I am finished')
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
