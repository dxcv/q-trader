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
import cfscrape

ex = ccxt.kraken({
    'apiKey': s.exchange_api_key,
    'secret': s.exchange_sk,
    'timeout': 20000,
    'session': cfscrape.create_scraper() # To avoid Cloudflare block
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
    ticker = ex.fetch_ticker(p.pair)
    return ticker['last']

def get_balance(asset=''):
    if asset == '': asset = p.currency
    balance = ex.fetch_balance()['free']
    return balance[asset]

def create_order(action, ordertype, volume, opt={}):
    params = (p.pair, volume)
    opt['ordertype'] = ordertype
    opt['leverage'] = p.leverage        
    params = params + (opt,)

    if action == 'Buy':
        order = ex.create_market_buy_order(*params)
    elif action == 'Sell':
        order = ex.create_market_sell_order(*params)
    
    result = fetch_order(order['id'])
    print('***** '+ordertype+' Order Created *****')
    print(result)

    return result

def wait_order(order_id):
    while len(list(filter(lambda t: t['id'] == order_id, ex.fetchOpenOrders(p.pair)))) > 0:
        time.sleep(p.order_wait)

def fetch_order(order_id):
    result = {}
    for rec in ex.fetchOpenOrders(p.pair):
        if rec['id'] == order_id: result = rec
    
    return result     

#def get_order_price(order_type):
#    orders = ex.fetchClosedOrders(p.pair)
#    return orders[0]['info']['price']

def market_order(action, volume=-1):
    if volume == -1: volume = p.order_size
    order = create_order(action, 'market', volume)
    # Wait till order is executed
    wait_order(order['id'])

    trade = ex.fetch_my_trades(p.pair)[-1]
    print('***** Order Executed *****')
    print(trade)

    # Get new balance
    fee = trade['fee']['cost']
    cost = trade['cost']
    size = trade['amount']
    price = round((cost + fee) / size, 4)
    balance = get_balance()
        
    result = {'size': size,
              'price': price,
              'fee': fee,
              'trade': trade,
              'balance': balance
              }

    return result

# Place Stop Loss Order
def sl_order(action):
    opt = {}
    opt['price'] = '#'+str(p.stop_loss * 100)+'%'
    order = create_order(action, 'stop-loss', p.order_size, opt)
    return 'SL: '+str(order['info']['descr']['price'])

# Place Take Profit Order
def tp_order(action):
    opt = {}
    opt['price'] = '#'+str(p.take_profit * 100)+'%'
    order = create_order(action, 'take-profit', p.order_size, opt)
    return 'TP: '+str(order['info']['descr']['price'])
        
def has_orders(types):
    for order in ex.fetchOpenOrders(p.pair):
        if order['type'] in types: return True
    return False

def has_sl_order():
    return has_orders(['stop-loss'])
    
def has_tp_order():
    return has_orders(['take-profit'])

def cancel_orders(types):
    for order in ex.fetchOpenOrders(p.pair):
        if order['type'] in types:
            print("Cancelling Order:")
            print(order)
            ex.cancelOrder(order['id'])    

def cancel_sl():
    cancel_orders(['stop-loss'])

def cancel_tp():
    cancel_orders(['take-profit'])

def test_order():
    # TP for Sell Position
    ex.create_market_buy_order('ETH/USD', 0, 
                                       { 
                                        'ordertype': 'take-profit',
                                        'price': '#3%',
                                        'leverage': 2
                                        }
                                       )
    
    
    # TP for Buy Position
    ex.create_market_sell_order('ETH/USD', 0.02,
                                       { 
                                        'ordertype': 'take-profit',
                                        'price': '#3%'
                                        }
                                       )
    
def test_order1():
    p.load_config('ETHUSDNN')
    p.order_size = 0.02
    
    # Buy
    market_order('Buy')
    market_order('Buy', sl=True)
    ex.fetch_balance()['free']
    # Close SL Order
    cancel_orders()
    # Sell
    market_order('Sell')
    ex.fetch_balance()['free']
    
    # Sell short
    market_order('Sell')
    # Close SL Order
    cancel_orders()
    # Close Short
    market_order('Buy')
    
    sl_order('Buy')

    ex.fetchOpenOrders()
    
    orders = ex.fetchClosedOrders('ETH/USD')
    order = orders[0]
