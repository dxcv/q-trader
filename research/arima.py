#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:16:48 2019

@author: igor
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

ds = pickle.load(open('data/ETHUSDNN/price.pkl', "rb" ))
ds = ds.set_index('date')
ds['DR'] = ds['close']/ds['close'].shift(1)
ds = ds.dropna()


# Check stationarity
result = adfuller(ds['DR'])
print(result[0], result[1], result[4])

result = adfuller(np.log(ds['DR']))
print(result[0], result[1], result[4])

# Fit ARMA model
model = ARMA(ds['DR'], order=(5, 5))
result = model.fit()
print(result.summary())

# Fit ARMAX model
model = ARMA(ds['DR'].to_numpy(), order=(5,5), exog=ds['RSI'].to_numpy())
results = model.fit()
print(results.summary())

# Fit SARIMAX model
model = SARIMAX(ds['close'], order=(2,2,2), freq='D')
results = model.fit()

# Generate predictions ******************************************************
one_step_forecast = results.get_prediction(start=-30)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean

# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower DR']
upper_limits = confidence_intervals.loc[:,'upper DR']

# Print best estimate  predictions
print(mean_forecast)

# plot the observed data
plt.plot(ds.index, ds['DR'], label='observed')

# plot your mean  predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, 
               upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('DR')
plt.legend()
plt.show()

# Generate dynamic predictions ******************************************************
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = dynamic_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = dynamic_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate predictions
print(mean_forecast)

# plot the observed data
plt.plot(ds.index, ds['close'], label='observed')

# plot your mean forecast
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()

# Forecast Future ******************************************************

# Create ARIMA(2,1,2) model
arima = SARIMAX(ds['close'], order=(2,2,2))

# Fit ARIMA model
arima_results = arima.fit()

# Make ARIMA forecast of next 10 values
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean

# Print forecast
print(arima_value_forecast)

# ACF and PACF ******************************************************
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = np.log(ds.DR)

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(df, lags=10, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(df, lags=10, zero=False, ax=ax2)

plt.show()

# AIC and BIC ******************************************************
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over q values from 0-2
    for q in range(3):
        try:
          	# create and fit ARMA(p,q) model
            model = SARIMAX(df, order=(p,0,q))
            results = model.fit()
            # Append order and results tuple
            order_aic_bic.append((p, q, results.aic, results.bic))
        except:
            order_aic_bic.append((p, q, None, None))

# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, columns=['p', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
print(order_df.sort_values('AIC'))

# Print order_df in order of increasing BIC
print(order_df.sort_values('BIC'))        

# Best p=4, q=5

# Mean absolute error ******************************************************
# Fit model
df = np.log(ds.DR)

model = SARIMAX(df, order=(2,0,2))
results = model.fit()
print(results.summary())

# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))

# Print mean absolute error
print(mae)

results.plot_diagnostics()
plt.show()

# Seasonal Time Series ******************************************************
# Import seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform additive decomposition
decomp = seasonal_decompose(ds['close'], freq=30)

# Plot decomposition
decomp.plot()
plt.show()

# Subtract the rolling mean
df = ds['close'] - ds['close'].rolling(100).mean()
# Drop the NaN values
df = df.dropna()

# Create figure and subplots
fig, ax1 = plt.subplots()

# Plot the ACF
plot_acf(df, lags=50, zero=False, ax=ax1)

# Show figure
plt.show()

df = ds['close']

# Create a SARIMAX model
model = SARIMAX(df, order=(2,2,2), seasonal_order=(2,1,2,30))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())

# Automated Model Search  ******************************************************
import pmdarima as pm

df = ds['Price_Rise']

# Create auto_arima model
model1 = pm.auto_arima(df,
#                      trend = 'c',
#                      seasonal=True, m=30,
                      d=1, 
#                      D=1, 
                 	  max_p=10, max_q=10,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model1.summary())

# 0,1,1

model = SARIMAX(df, order=(1,1,1))
results = model.fit()
print(results.summary())

# Calculate the mean absolute error from residuals
res = results.resid
print(np.mean(np.abs(res)))
sum(abs(res) < 0.5)/len(res)

results.plot_diagnostics()
plt.show()

# Save / Load Model ******************************************************
# Import joblib
import joblib

# Set model name
filename = 'model.pkl'

# Pickle it
joblib.dump(model,filename)

# Load Model
loaded_model = joblib.load(filename)

# Update Model with new data
loaded_model.update(df)

model = SARIMAX(ds['Price_Rise'], order=(2,1,2))
results = model.fit()
print(results.summary())
print(np.mean(np.abs(results.resid)))
res = results.resid
print(sum(abs(res) < 0.5)/len(res))