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
        if dt.datetime.today() > signal['end']:
            send('Signal has expired. Waiting for new one ...')
            time.sleep(p.sleep_interval)
        else:
            no_signal = False

    return signal

def send_results(res, msg):
    send(msg+' of '+str(res['size'])+' '+res['pair'])
    send('New Balance: ' + str(res['balance']))

def execute(conf):
    signal = get_signal(conf)
    ticker = conf[0:3]+'/'+conf[3:6]
    send('Market: '+ticker, True)
 
    if not signal['new']:
        send('No action today', True)
    elif signal['signal'] == 'Buy':
        send('Buy! Buy! Buy!', True)
        if p.short:
            res = x.market_order('Buy', True)
            send_results(res, 'Closed Short Position')
        res = x.market_order('Buy')
        send_results(res, 'Opened Long Position')
    elif signal['signal'] == 'Sell':
        send('Sell! Sell! Sell!', True)
        res = x.market_order('Sell')
        send_results(res, 'Closed Long Position')
        if p.short:
            res = x.market_order('Sell', True)
            send_results(res, 'Opened Short Position')

try:
    send('Hi Master!')
    execute('ETHUSDNN')
except Exception as e:
    send('An error has occured. Please investigate!')
    send(e)
finally:
    send('Have a good day!')
    t.cleanup()
