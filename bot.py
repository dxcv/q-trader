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
    send(msg+' of '+str(res['amount'])+' '+p.pair+' with price '+str(res['price']))


def execute(s, s0):
    action = s['action']
    prev_action = s0['action']
    
    # Check position on exchange
    is_open = x.has_open_position()
    
    if not is_open:
        if p.stop_loss < 1 and not x.has_sl_order():
            send('Stop Loss triggered!')
        
        if p.take_profit > 0 and not x.has_tp_order():
            send('Take Profit triggered!')
    
    # Cancel any open SL and TP orders
    x.cancel_orders()
    
    # Close position if it is open and new trade or SL / TP triggered 
    if is_open and (s['new_trade'] or s['tp'] or s['sl']):
        res = x.close_position(prev_action)
        send_results(res, 'Closed '+prev_action+' Position')
        is_open = False
    
    # Do not open new trade if SL or TP already triggered for current day
    if s['tp'] or s['sl']:
        return

    if not is_open and (action == 'Buy' or action == 'Sell' and p.short):
        res = x.open_position(action)
        send_results(res, 'Opened '+action+' Position')
        is_open = True

    """ SL/TP can only be set AFTER order is executed if margin is not used """
    if is_open:
        if p.take_profit > 0: x.take_profit(action, s['tp_price'])
        if p.stop_loss < 1: x.stop_loss(action, s['sl_price'])
            
def run_model(conf):
        s = get_signal(conf)
        s0 = nn.get_signal(-2)
     
        # Send signal
        send(nn.get_signal_str(s), True)
        
        if p.execute: 
            try:
                execute(s, s0)
                send('Balance: '+str(x.get_balance()))
            except Exception as e:
                send('An error has occured. Please investigate!')
                send(e)
        
def test_execute():
    p.load_config('ETHUSDNN')
    p.order_size = 0.02
    s = {}
    s0 = {}
    s['action'] = 'Buy'
    s0['action'] = 'Buy'
    s['new_trade'] = False
    s['sl'] = False
    s['tp'] = False
    s['sl_price'] = 100
    s['tp_price'] = 200
    
    execute(s, s0)


send('Current Model:', True)
run_model('ETHUSDNN')

send('New Model:', True)
run_model('ETHUSDLSTM')
#
t.cleanup()

