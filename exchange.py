#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kraken API
# https://github.com/dominiktraxl/pykrakenapi

# CCXT API
# https://github.com/ccxt/ccxt/wiki/Manual#overriding-unified-api-params

"""
Created on Mon Dec 25 18:06:07 2017

@author: imonahov
"""
import ccxt
import time
import params as p
import mysecrets as s

ex = ccxt.kraken({
#    'verbose': True,    
    'apiKey': s.exchange_api_key,
    'secret': s.exchange_sk,
    'timeout': 20000,
#    'session': cfscrape.create_scraper(), # To avoid Cloudflare block => still fails with 520 Origin Error
    'enableRateLimit': True,
    'rateLimit': 1000 # Rate Limit set to 1 sec to avoid issues
})

markets = ex.load_markets()

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

# Returns day open price
def get_price():
    ticker = ex.fetch_ticker(p.pair)
    return ticker['open']

def get_balance(asset=''):
    if asset == '': asset = p.currency
    balance = ex.fetch_balance()['free']
    return balance[asset]

def get_balance_str():
    balance = ex.fetch_balance()['free']
    return p.currency+': '+str(balance[p.currency])+', '+p.ticker+': '+str(balance[p.ticker])

def create_order(side, amount=0, price=0, ordertype='', leverage=1, wait=True):
    params = {}
    if ordertype == '': ordertype = p.order_type
    if leverage > 1: params['leverage'] = leverage
    if price == 0 and ordertype == 'limit': params['price'] = '#0%'

    order = ex.create_order(p.pair, ordertype, side, amount, price, params)    
    order = ex.fetchOrder(order['id'])
    print('***** Order Created *****')
    print(order)

    # Wait till order is executed
    if wait: order = wait_order(order['id'])

    return order

def fetchOrder(order_id):
    order = {}
    try:
        order = ex.fetchOrder(order_id)
    except Exception as e:
        print(e)
    
    return order

def wait_order(order_id):
    print('Waiting for order '+order_id+' to be executed ...')
    while True:
        order = fetchOrder(order_id)
        if order != {} and order['status'] == 'closed':
            print('***** Order Executed *****')
            print(order)
            return order
        time.sleep(p.order_wait)

#def get_order_price(order_type):
#    orders = ex.fetchClosedOrders(p.pair)
#    return orders[0]['info']['price']

def get_pos_size(action):
    size = p.order_size
    if action == 'Sell' and p.max_short > 0: size = min(p.order_size, p.max_short)
    
    return size

# Returns Order Size based on order_pct parameter
def get_order_size(action, price=0):
    if p.order_size > 0: return get_pos_size(action)
    
    # Calculate position size based on available balance
    if price == 0: price = get_price()
    balance = get_balance()
    amount = balance * p.order_pct
    size = p.truncate(amount/price, p.order_precision)
    if p.short and p.max_short > 0 and action == 'Sell': size = min(p.max_short, size)
    return size

def close_position(action, amount=0, price=0, ordertype='', wait=True):
    res = {}
    if ordertype == '': ordertype = p.order_type
    if amount == 0 and p.order_size > 0: amount = get_pos_size(action)
    
    if action == 'Sell':
        res = create_order('buy', amount, price, ordertype, p.leverage, wait)
    elif action == 'Buy':
        if amount == 0: amount = get_balance(p.ticker)
        res = create_order('sell', amount, price, ordertype, 1, wait)

    return res

def open_position(action, amount=0, price=0, ordertype='', wait=True):
    res = {}
    if amount == 0: amount = get_order_size(action, price)

    if action == 'Sell':
        res = create_order('sell', amount, price, ordertype, p.leverage, wait)
    elif action == 'Buy':
        res = create_order('buy', amount, price, ordertype, 1, wait)

    return res
        
def take_profit(action, price):
    close_position(action, ordertype='take-profit', price=price, wait=False)

def stop_loss(action, price):
    close_position(action, ordertype='stop-loss', price=price, wait=False)

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

def get_position():
    if get_balance(p.ticker) > p.min_equity: return 'Buy'
    if not p.short: return 'Sell'

    # Check short position
    res = ex.privatePostOpenPositions()
    if len(res['result']) > 0: return 'Sell'
    
    return 'Cash'

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

def test_order1():
    p.load_config('ETHUSDNN')
    p.order_size = 0.02
    # Print available API methods
    print(dir(ex))
    
    # Buy
    ex.fetch_balance()['free']
    # Close SL Order
    cancel_orders()
    
    ex.fetchOpenOrders()
    
    ex.fetchClosedOrders('ETH/USD')

    # Get Open Positions
    ex.privatePostOpenPositions()

    # Limit Order with current price
    create_order('Buy', 'limit', 0.02, {'price':'+0%'})
    
    ex.createOrder('ETH/USD', 'market', 'buy', 0.02)

def test_order2():
    p.load_config('ETHUSDNN')

    # Create Market Order
    ex.createOrder('ETH/USD', 'market', 'buy', 0.02)
    ex.createOrder('ETH/USD', 'market', 'sell', 0.02)
    ex.createOrder('ETH/USD', 'market', 'buy', 0.02, 0) # Price is ignored

    # Create Limit Order for fixed price
    ex.createOrder('ETH/USD', 'limit', 'buy', 0.02, 100)
    # Create Limit Order for -1% to market price
    ex.createOrder('ETH/USD', 'limit', 'buy', 0.02, 0, {'price':'-1%'})

    # Fetch Open Orders
    orders = ex.fetchOpenOrders()
    # Order Size
    orders[0]['amount']

    ex.fetchBalance()['ETH']

def test_order3():
    p.load_config('ETHUSDNN')
    p.order_size = 0.02
    p.order_wait = 10
    open_position('Buy')
    print(get_balance())
    
    res = take_profit('Buy', 200)
    res = stop_loss('Buy', 100)
    res = close_position('Buy', wait=False)
    get_balance('ETH')
    ex.fetchOpenOrders()
    cancel_sl()
    cancel_tp()
    cancel_orders()
    get_price()

    res = ex.privatePostOpenPositions()
    len(res['result'])
    open_position('Sell')
    close_position('Sell', wait=False)
    ex.fetchOpenOrders()
    get_price()
        