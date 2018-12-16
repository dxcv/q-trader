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
    send(msg+' of '+str(res['size'])+' '+res['pair']+' with price '+str(res['price']))
    send('New Balance: ' + str(res['balance']))

def execute(conf):
    s = get_signal(conf)
 
    send(p.ticker+'/'+p.currency, True)
    # Send details about previous and current positions
    send('Yesterday: ' + nn.get_signal_str(-2), True)
    send('Today: ' + nn.get_signal_str(), True)
    if p.execute: 
        # Cancel SL Order
        x.cancel_orders()
        if s['new']:
            if s['action'] == 'Buy':
                if p.short:
                    res = x.market_order('Buy', leverage=True)
                    send_results(res, 'Closed Short Position')
                res = x.market_order('Buy')
                send_results(res, 'Opened Long Position')
            elif s['action'] == 'Sell':
                res = x.market_order('Sell')
                send_results(res, 'Closed Long Position')
                if p.short:
                    res = x.market_order('Sell', leverage=True)
                    send_results(res, 'Opened Short Position')

        # Set Stop Loss for current position (new or old)
        if s['action'] == 'Buy':
            res = x.stop_loss_order('Sell')
            send(res)
        elif s['action'] == 'Sell' and p.short:
            res = x.stop_loss_order('Buy', leverage=True)
            send(res)
 

def run_model(conf):
    try:
        execute(conf)
    except Exception as e:
        send('An error has occured. Please investigate!')
        send(e)
    
run_model('ETHUSDNN')
t.cleanup()
