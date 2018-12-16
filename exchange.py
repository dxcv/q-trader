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
    ticker = ex.fetch_ticker(p.ticker + '/' + p.currency)
    return ticker['last']

def get_balance():
    balance = ex.fetch_balance()['free']
#    print('***** Current Balance *****')
#    print(balance)
    return balance

def create_order(action, ordertype='market', pair='', size = 0, leverage=False):
    if pair == '': pair = p.ticker+'/'+p.currency
    if size == 0: size = p.order_size
    
    params = (pair, size)
    opt = {}

    if leverage: opt['leverage'] = p.leverage
    if ordertype == 'stop-loss':
        opt['ordertype'] = ordertype
        opt['price'] = '#'+str(p.stop_loss * 100)+'%'
        
    if len(opt) > 0: params = params + (opt,)

    if action == 'Buy':
        order = ex.create_market_buy_order(*params)
    elif action == 'Sell':
        order = ex.create_market_sell_order(*params)
    
    return order

def wait_order(order_id):
    while len(list(filter(lambda t: t['id'] == order_id, ex.fetchOpenOrders()))) > 0:
        time.sleep(p.order_wait)

def fetch_order(order_id):
    result = {}
    for rec in ex.fetchOpenOrders():
        if rec['id'] == order_id: result = rec
    
    return result     
    
def market_order(action, pair='', size = 0, leverage=False):
    order = create_order(action, 'market', pair, size, leverage)
    # Wait till order is executed
    wait_order(order['id'])

    trade = ex.fetch_my_trades(pair)[-1]
    print('***** Order Executed *****')
    print(trade)

    # Get new balance
    fee = trade['fee']['cost']
    cost = trade['cost']
    size = trade['amount']
    price = round((cost + fee) / size, 4)
    balance = dict((k, get_balance()[k]) for k in (p.ticker, p.currency))
        
    result = {'pair': pair,
              'size': size,
              'price': price,
              'fee': fee,
              'trade': trade,
              'balance': balance
              }

    return result

def stop_loss_order(action, pair='', size = 0, leverage=False):
    order = create_order(action, 'stop-loss', pair, size, leverage)
    result = fetch_order(order['id'])
    print('***** SL Order Created *****')
    print(result)
    return result['info']['descr']['order']
        
def cancel_orders(pair='', types=['stop-loss']):
    if pair == '': pair = p.ticker+'/'+p.currency
    orders = ex.fetchOpenOrders(pair)
    for order in orders:
        if order['type'] in types:
            print("Cancelling Order:")
            print(order)
            ex.cancelOrder(order['id'])    

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
    
    # Sell short with sl = 1000
    market_order('Sell', sl = 1000, leverage=True)
    # Close SL Order
    cancel_orders()
    # Close Short
    market_order('Buy', leverage=True)
    
    stop_loss_order('Buy', 'ETH/USD', 0.02)

    ex.fetchOpenOrders()