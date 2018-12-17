#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:47:41 2018

@author: igor
"""

import nn
import tele as t
import exchange as x
import datetime as dt
import time
import params as p

def send(msg, public=False):
    print(msg)
    t.send_msg(str(msg), public)

def get_signal(conf):
    no_signal = True

    while no_signal:
        nn.runNN(conf)
        signal = nn.get_signal()
        if dt.datetime.today() > signal['close_ts']:
            send('Signal has expired. Waiting for new one ...')
            time.sleep(p.sleep_interval)
        else:
            no_signal = False

    return signal

def send_results(res, msg):
    send(msg+' of '+str(res['size'])+' '+p.pair+' with price '+str(res['price']))
    send('Balance: ' + str(res['balance']))

def execute(conf):
    s = get_signal(conf)
 
    send(p.pair, True)
    # Send details about previous and current positions
    send('Yesterday: ' + nn.get_signal_str(-2), True)
    send('Today: ' + nn.get_signal_str(), True)
    if p.execute:
        action = s['action']
        is_open = True
# FIXME: triggering both SL and TP should be handled / avoided
        if x.has_sl_order():
            x.cancel_sl()
        else:
            is_open = False
            send('Stop Loss triggered!')
            send('Balance: '+str(x.get_balance))
        
        if x.has_tp_order():
            x.cancel_tp()
        else:
            is_open = False
            send('Take Profit triggered!')
            send('Balance: '+str(x.get_balance))
        
        # Close position if signal has changed and it is still open
        if s['new_signal'] and is_open:
            if action == 'Buy' and p.short:
                res = x.market_order('Buy', 0)
                send_results(res, 'Closed Short Position')
            elif action == 'Sell':
                res = x.market_order('Sell', 0)
                send_results(res, 'Closed Long Position')
            is_open = False
        
        if not is_open:
            if action == 'Buy':
                res = x.market_order('Buy')
                send_results(res, 'Opened Long Position')
            elif action == 'Sell' and p.short:
                res = x.market_order('Sell')
                send_results(res, 'Opened Short Position')

        # Set Stop Loss and Take Profit for current position
        if action == 'Buy':
            send(x.sl_order('Sell'))
            send(x.tp_order('Sell'))
        elif action == 'Sell' and p.short:
            send(x.sl_order('Buy'))
            send(x.tp_order('Buy'))
 

def run_model(conf):
    try:
        execute(conf)
    except Exception as e:
        send('An error has occured. Please investigate!')
        send(e)
    
run_model('ETHUSDNN')
t.cleanup()
