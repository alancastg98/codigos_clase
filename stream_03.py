# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:10:54 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


# input
ric = 'RDSa.AS' # DBK.DE ^IXIC MXN=X ^STOXX ^S&P500 ^VIX

# get market data
# remember to modify the path to match your own directory
path = 'C:\\Users\\alanj\\OneDrive\\Documents\\GitHub\\homework-0-alan-castg98\\data\\' + ric + '.csv' 
table_raw = pd.read_csv(path)

# create table of returns
t = pd.DataFrame()
t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
t['close'] = table_raw['Close']
t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1)
t['return_close'] = t['close']/t['close_previous'] - 1
t = t.dropna()
t = t.reset_index(drop=True)

# plot timeseries of price
plt.figure()
plt.plot(t['date'],t['close'])
plt.title('Time series real prices ' + ric)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# input for Jarque-Bera test
x = t['return_close'].values # returns as array
x_str = 'Real returns ' + ric # label e.g. ric
x_size = len(x) # size of returns


### Recycled code from stream_02.py ###

# compute "risk metrics"
x_mean = np.mean(x)
x_std = np.std(x) # volatility
x_skew = skew(x)
x_kurt = kurtosis(x) # excess kurtosis
x_sharpe = x_mean / x_std * np.sqrt(252) # annualised
x_var_95 = np.percentile(x,5)
x_cvar_95 = np.mean(x[x <= x_var_95])
jb = x_size/6*(x_skew**2 + 1/4*x_kurt**2)
p_value = 1 - chi2.cdf(jb, df=2)
is_normal = (p_value > 0.05) # equivalently jb < 6

# print metrics
round_digits = 4
str1 = 'mean ' + str(np.round(x_mean,round_digits))\
    + ' | std dev ' + str(np.round(x_std,round_digits))\
    + ' | skewness ' + str(np.round(x_skew,round_digits))\
    + ' | kurtosis ' + str(np.round(x_kurt,round_digits))\
    + ' | Sharpe ratio ' + str(np.round(x_sharpe,round_digits))
str2 = 'VaR 95% ' + str(np.round(x_var_95,round_digits))\
    + ' | CVaR 95% ' + str(np.round(x_cvar_95,round_digits))\
    + ' | jarque_bera ' + str(np.round(jb,round_digits))\
    + ' | p_value ' + str(np.round(p_value,round_digits))\
    + ' | is_normal ' + str(is_normal)
    
# plot histogram
plt.figure()
plt.hist(x,bins=100)
plt.title('Histogram ' + x_str)
plt.xlabel(str1 + '\n' + str2)
plt.show()