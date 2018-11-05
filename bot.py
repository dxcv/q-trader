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

def send_results(res, msg, pnl=False):
    send(msg+' of '+str(res['size'])+' '+res['pair'])
    # PnL calculation needs to be based on original balance when position was opened  
    if pnl:  
        send('with '+('profit' if res['cash_diff'] > 0 else 'loss')+' of '
             +str(round(res['cash_diff'], 2))+' '+p.currency)
    send('New Balance: ' + str(res['balance']))

def execute(conf):
    send('I am started')
    signal = get_signal(conf)
 
    if signal['hold']:
        send('Hold - no action is required')
    elif signal['signal'] == 'Buy':
        send('Received Buy Signal')
        res = x.market_order('Buy', True)
        send_results(res, 'Closed Short Position')
        res = x.market_order('Buy')
        send_results(res, 'Opened Long Position')
    elif signal['signal'] == 'Sell':
        send('Received Sell Signal')
        res = x.market_order('Sell')
        send_results(res, 'Closed Long Position')
        res = x.market_order('Sell', True)
        send_results(res, 'Opened Short Position')

try:
    execute('ETHUSDNN')
except Exception as e:
    send('An error occured')
    send(e)
finally:
    send('I am finished')
    t.cleanup()
