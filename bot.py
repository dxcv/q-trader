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
    while True:
        nn.runModel(conf)
        signal = nn.get_signal()
        if dt.datetime.today() > signal['close_ts']:
            send('Signal has expired. Waiting for new one ...')
            time.sleep(p.sleep_interval)
        else:
            return signal

def send_results(res, msg):
    send(msg+' of '+str(res['filled'])+' '+p.pair+' with price '+str(res['average']))
    send('Balance: '+x.get_balance_str())

def execute(s):
    action = s['action']
    position = x.get_position()    
    is_open = (position == 'Buy' or position == 'Sell' and p.short)
    
    if p.stop_loss and not x.has_sl_order(): send('SL is not set!')
    if p.take_profit < 1 and not x.has_tp_order(): send('TP is not set!')
    
    # Cancel any open orders
    x.cancel_orders()
    
    # Close position if new trade
    if is_open and action != position:
        res = x.close_position(position)
        send_results(res, 'Closed '+position+' Position')
        is_open = False
    
    if not is_open and (action == 'Buy' or action == 'Sell' and p.short):
        res = x.open_position(action)
        send_results(res, 'Opened '+action+' Position')
        is_open = True

    """ SL/TP can only be set AFTER order is executed if margin is not used """
    if p.stop_loss and s['sl_price'] > 0: 
        x.stop_loss(action, s['sl_price'])
        send('SL set at '+str(s['sl_price']))

    if is_open and p.take_profit < 1: 
        x.take_profit(action, s['tp_price'])
        send('TP set at '+str(s['tp_price']))
            
def run_model(conf):
        s = get_signal(conf)
     
        # Send signal
        send(nn.get_signal_str(s), True)
        
        if p.execute: 
            try:
                execute(s)
            except Exception as e:
                send('An error has occured. Please investigate!')
                send(e)
        
def test_execute():
    p.load_config('ETHUSDNN')
    p.order_size = 0.02
    s = {}
    s['action'] = 'Buy'
    s['new_trade'] = False
    s['sl'] = False
    s['tp'] = False
    s['sl_price'] = 100
    s['tp_price'] = 200
    
    execute(s)


run_model('ETHUSDNN')

t.cleanup()
