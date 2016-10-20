# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:15:51 2016

@author: bagas
"""
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from thesis import gp as gp2
from sklearn import gaussian_process

style.use('seaborn-whitegrid')

data_folder = '../data/weather/'

#data = pd.read_csv(data_folder + 'Capelle.csv',
#                   parse_dates=['DateTime'],
#                   index_col=['DateTime'])
                 
#%% wind speed data
wind_speed = data['WindSpd_{Avg}']
y = wind_speed['2016/1/1':'2016/1/7'].astype('float')
x = np.arange(y.size)

y_ = wind_speed['2016/1/8']
x_ = np.arange(y.size, y.size + y_.size)

param = {'lambda': 5, 'sigma': 5}

gp_regressor = gp2.GaussianProcess(param=param)
gp_regressor.fit(x,y)
y_pred, cov = gp_regressor.predict(x_)

plt.figure(2)
plt.plot(x_, y_)
plt.plot(x_, y_pred)
plt.legend(['Actual', 'GP Regression'])

#%% use sklearn GP implementation
skgp = gaussian_process.GaussianProcess()
skgp.fit(x,y)
y_pred2, sigma_pred = skgp.predict(x_)

plt.figure(2)
plt.plot(x_, y_)
plt.plot(x_, y_pred)
plt.plot(x_, y_pred2)
plt.legend(['Actual', 'GP Regression', 'sklearn GP'])