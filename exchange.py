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
    
    result = ex.fetchOrder(order['id'])
    print('***** '+ordertype+' Order Created *****')
    print(result)

    return result

def wait_order(order_id):
    print('Waiting for order '+order_id+' to be executed ...')
    while ex.fetchOrder(order_id)['status'] != 'closed':
        time.sleep(p.order_wait)
    order = ex.fetchOrder(order_id)
    print('***** Order Executed *****')
    print(order)
    return order

#def get_order_price(order_type):
#    orders = ex.fetchClosedOrders(p.pair)
#    return orders[0]['info']['price']

def execute_order(action, ordertype='', volume=-1, price=0, wait=False):
    opt = {}
    if ordertype == '': ordertype = p.order_type
    if volume == -1: volume = p.order_size
    if price == 0: price = '#0%' 
    if ordertype == 'limit': opt = {'price': price}

    order = create_order(action, ordertype, volume, opt)
    # Wait till order is executed
    if wait: order = wait_order(order['id'])

    fee = order['fee']['cost']
    size = order['amount']
    price = order['price']
        
    result = {'size': size, 'price': price, 'fee': fee}
    return result

# Place Stop Loss Order
def sl_order(action):
    opt = {}
    opt['price'] = '#'+str(p.stop_loss * 100)+'%'
    order = create_order(action, 'stop-loss', p.order_size, opt)
    return action+' Stop Loss at '+str(order['info']['descr']['price'])

# Place Take Profit Order
def tp_order(action):
    opt = {}
    opt['price'] = '#'+str(p.take_profit * 100)+'%'
    order = create_order(action, 'take-profit', p.order_size, opt)
    return action+' Take Profit at '+str(order['info']['descr']['price'])
        
def has_orders(types=[]):
    if types == []: types = [p.order_type]
    for order in ex.fetchOpenOrders(p.pair):
        if order['type'] in types: return True
    return False

def wait_orders(types=[]):
    if types == []: types = [p.order_type]
    for order in ex.fetchOpenOrders(p.pair):
        if order['type'] in types: wait_order(order['id'])

def has_sl_order():
    return has_orders(['stop-loss'])
    
def has_tp_order():
    return has_orders(['take-profit'])

def cancel_orders(types=[]):
    for order in ex.fetchOpenOrders(p.pair):
        if types == [] or order['type'] in types:
            print("Cancelling Order:")
            print(order)
            ex.cancelOrder(order['id'])    

def cancel_sl():
    cancel_orders(['stop-loss'])

def cancel_tp():
    cancel_orders(['take-profit'])

def close_position(pos_type):
    res = ''
    if pos_type == 'Sell':
        res = execute_order('Buy', volume=0)
    elif pos_type == 'Buy':
        res = execute_order('Sell', volume=0)
    return res

def test_order1():
    p.load_config('ETHUSDNN')
    p.order_size = 0.02
    # Print available API methods
    print(dir(ex))
    
    # Buy
    execute_order('Buy')
    ex.fetch_balance()['free']
    # Close SL Order
    cancel_orders()
    # Sell
    execute_order('Sell')
    ex.fetch_balance()['free']
    
    ex.fetchOpenOrders()
    
    orders = ex.fetchClosedOrders('ETH/USD')
    order = orders[0]

    # Get Open Positions
    ex.privatePostOpenPositions()

    # Limit Order with current price
    create_order('Buy', 'limit', 0.02, {'price':'0%'})
    
    ex.createOrder('ETH/USD', 'market', 'buy', 0.02)
    
    execute_order('Sell', 'market', volume=0.02)
    