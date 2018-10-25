#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:47:41 2018

@author: igor
"""

import ccxt
import time

def run_bot():
    ex = ccxt.kraken()

    while True:
        ethusd = ex.fetch_ticker('ETH/USD')
        print('Current Price: ' + str(ethusd['last']))
        time.sleep(60)
