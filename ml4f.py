#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:51:45 2019

@author: igor
"""

import params as p
import datalib as dl
import matplotlib.pyplot as plt

# Inspired by:
# Machine Learning for Finance in Python
# https://campus.datacamp.com/courses/machine-learning-for-finance-in-python

# Explore the data with some EDA

conf = 'ETHUSDNN'
p.load_config(conf)
ds = dl.load_price_data()

print(ds.head(5))  # examine the DataFrames

# Plot the close column for ETH
ds['close'].plot(label='ETH', legend=True)
plt.show()  # show the plot
plt.clf()  # clear the plot space

# Histogram of the daily price change percent of Adj_Close for LNG
ds['close'].pct_change().plot.hist(bins=50)
plt.xlabel('adjusted close 1-day percent change')
plt.show()
