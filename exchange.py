#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:06:07 2017

@author: imonahov
"""
import ccxt
import time
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

def get_balance():
    balance = ex.fetch_balance()['free']
#    print('***** Current Balance *****')
#    print(balance)
    return balance

def market_order(action, leverage=False):
    b1 = get_balance()
    pair = p.ticker+'/'+p.currency
        
    if action == 'Buy':
        if leverage:
#            print('Placing Market Buy Order with leverage 2')
            order = ex.create_market_buy_order(pair, p.order_size, {'leverage': 2})
        else:
#            print('Placing Market Buy Order')
            order = ex.create_market_buy_order(pair, p.order_size)
    elif action == 'Sell':
        if leverage:
#            print('Placing Market Sell Order with leverage 2')
            order = ex.create_market_sell_order(pair, p.order_size, {'leverage': 2})
        else:
#            print('Placing Market Sell Order')
            order = ex.create_market_sell_order(pair, p.order_size)

    print('***** Order Placed *****')
    print(order)

    # Wait till order is executed
    while len(ex.fetch_open_orders(symbol=pair)) > 0: time.sleep(p.order_wait)

    # Get new balances
    b2 = get_balance()    
    result = {'pair': pair, 'size': p.order_size, 
              'cash_diff': b2[p.currency] - b1[p.currency],
              'unit_diff': b2[p.ticker] - b1[p.ticker],
              'order': order,
              'balance': b2
              }

    return result

    