#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:06:07 2017

@author: imonahov
"""
import ccxt
import time
import math
import params as p
import secrets as s

ex = ccxt.kraken({
    'apiKey': s.exchange_api_key,
    'secret': s.exchange_sk
})


'''
hitbtc = ccxt.hitbtc({'verbose': True})
bitmex = ccxt.bitmex()
huobi  = ccxt.huobi()
exmo   = ccxt.exmo({
    'apiKey': 'YOUR_PUBLIC_API_KEY',
    'secret': 'YOUR_SECRET_PRIVATE_KEY',
})

hitbtc_markets = hitbtc.load_markets()

print(hitbtc.id, hitbtc_markets)
print(bitmex.id, bitmex.load_markets())
print(huobi.id, huobi.load_markets())

print(hitbtc.fetch_order_book(hitbtc.symbols[0]))
print(bitmex.fetch_ticker('BTC/USD'))
print(huobi.fetch_trades('LTC/CNY'))

print(exmo.fetch_balance())

# sell one ฿ for market price and receive $ right now
print(exmo.id, exmo.create_market_sell_order('BTC/USD', 1))

# limit buy BTC/EUR, you pay €2500 and receive ฿1  when the order is closed
print(exmo.id, exmo.create_limit_buy_order('BTC/EUR', 1, 2500.00))

# pass/redefine custom exchange-specific order params: type, amount, price, flags, etc...
exmo.create_market_buy_order('BTC/USD', 1, {'trading_agreement': 'agree'})
'''

'''
Executes Market Order on exchange
Example: Buy BTC with 100 EUR
order = market_order('buy', 'BTC', 'EUR', 100)
Example: Sell 0.0001 BTC
order = market_order('sell', 'BTC', 'EUR', 0.0001)
'''

def get_price():
    ticker = ex.fetch_ticker(p.ticker + '/' + p.currency)
    return ticker['last']

def market_order(action, amount, ticker=None, currency=None):
    action = action.lower()
    if not ticker: ticker = p.ticker
    if not currency: currency = p.currency
    balance = ex.fetch_balance()
    print('***** Current Balance *****')
    print(balance['free'])
    cash = balance['free'][currency]
    equity = balance['free'][ticker]

    lot = min(cash, amount) if action == 'buy' else min(equity, amount)  
    lot = math.trunc(lot*10**7)/10**7 # Leave only 7 decimals
    if action == 'buy' and lot < p.min_cash:
        print('No enough cash to place buy order: '+str(lot))
        return
    if action == 'sell' and lot < p.min_equity:
        print('No enough equity to place sell order: '+str(lot))
        return
    symbol = ticker+'/'+currency

    print('Market Order: action='+action+' symbol='+symbol+' funds='+str(lot))
    if action == 'buy':
        order = ex.create_market_buy_order(symbol, 0, params={'funds': str(lot)})
    else:
        order = ex.create_market_sell_order(symbol, lot)

    print('***** Order Placed *****')
    print(order)
    # Wait till order is executed
    while len(ex.fetch_open_orders(symbol=symbol)) > 0: time.sleep(p.order_wait)

#   Get new balances
    balance = ex.fetch_balance()
    print('***** New Balance *****')
    print(balance['free'])
    cash1 = balance['free'][currency]
    equity1 = balance['free'][ticker]
    cash_used = cash1 - cash
    equity_used = equity1 - equity

    return cash_used, equity_used


