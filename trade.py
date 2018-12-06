#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:43:21 2018

@author: igor
"""

import params as p
import pandas as pd
import datetime as dt

class Portfolio:
    def __init__(self, balance):
          self.cash = balance
          self.equity = 0.0
          self.short = 0.0
          self.total = balance    
    
    def upd_total(self):
        self.total = self.cash+self.equity+self.short


def buy_lot(pf, lot, short=False):
    if lot > pf.cash: lot = pf.cash
    pf.cash -= lot
    adj_lot = lot*(1-p.fee)
    if short: pf.short += adj_lot
    else: pf.equity += adj_lot
    
def sell_lot(pf, lot, short=False):
    if short:
        if lot > pf.short: lot = pf.short
        pf.short -= lot
    else:
        if lot > pf.equity: lot = pf.equity 
        pf.equity -= lot
    
    pf.cash = pf.cash + lot*(1-p.fee)

class TradeLog:
    def __init__(self):
        self.cash = p.start_balance
        self.equity = 0.0
        columns = ['date', 'action', 'cash', 'equity', 'price', 'cash_bal', 'equity_bal']
        self.log = pd.DataFrame(columns=columns)
    
    def log_trade(self, action, cash, equity):
        price = abs(cash)/abs(equity)
        self.cash += cash
        self.equity += equity
        row = [{'date': dt.datetime.now(),'action':action, 
            'cash':cash, 'equity':equity, 'price':price, 
            'cash_bal':self.cash, 'equity_bal':self.equity}]
        self.log = self.log.append(row, ignore_index=True)

