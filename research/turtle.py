#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:23:41 2019

@author: igor
"""

# https://www.quora.com/Does-the-Turtle-Trading-system-still-work

# To get closing price data  
from pandas_datareader import data as pdr  
import yfinance as yf  
yf.pdr_override()  
# Plotting graphs  
import matplotlib.pyplot as plt  
import seaborn  
# Data manipulation  
import numpy as np  
import pandas as pd  

def strategy_performance(stock_ticker):  
  #Get the data for the stock_ticker from yahoo finance.
  stock = pdr.get_data_yahoo(stock_ticker, start="2009-01-01", end="2017-10-01")
  
  # 5-days high  
  stock['high'] = stock.Close.shift(1).rolling(window=5).max()  
  # 5-days low  
  stock['low'] = stock.Close.shift(1).rolling(window=5).min()  
  # 5-days mean  
  stock['avg'] = stock.Close.shift(1).rolling(window=5).mean()  

  # Entry Rules
  stock['long_entry'] = stock.Close > stock.high  
  stock['short_entry'] = stock.Close < stock.low  
  
  # Exit Rules
  stock['long_exit'] = stock.Close < stock.avg  
  stock['short_exit'] = stock.Close > stock.avg 
  
  # Positions
  stock['positions_long'] = np.nan  
  stock.loc[stock.long_entry,'positions_long']= 1  
  stock.loc[stock.long_exit,'positions_long']= 0  
  stock['positions_short'] = np.nan  
  stock.loc[stock.short_entry,'positions_short']= -1  
  stock.loc[stock.short_exit,'positions_short']= 0  
  stock['Signal'] = stock.positions_long + stock.positions_short  
  stock = stock.fillna(method='ffill')
  
  # Strategy Returns
  daily_log_returns = np.log(stock.Close/stock.Close.shift(1))  
  daily_log_returns = daily_log_returns * stock.Signal.shift(1)  
  
  # Plot the distribution of 'daily_log_returns'  
  print (stock_ticker)  
  daily_log_returns.hist(bins=50)  
  plt.show() 
  
  return daily_log_returns.cumsum() 

# Create a portfolio of stocks and calculate the strategy performance for each stock.
portfolio = ['AAPL','KMI','F']  
cum_daily_return = pd.DataFrame()  
for stock in portfolio:  
  cum_daily_return[stock] = strategy_performance(stock)  
# Plot the cumulative daily returns  
print ("Cumulative Daily Returns")
cum_daily_return.plot()  
# Show the plot  
plt.show()  