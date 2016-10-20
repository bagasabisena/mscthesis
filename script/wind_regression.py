# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:15:11 2016

@author: bagas
"""
import pandas as pd
import numpy as np
import gp
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-whitegrid')

data_folder = '../data/weather/'

data = pd.read_csv(data_folder + 'Capelle.csv',
                   parse_dates=['DateTime'],
                   index_col=['DateTime'])
                 
#%% wind speed data
wind_speed = data['WindSpd_{Avg}']
day = wind_speed['2016/1/1'].astype('float')

y_ = day.values
x_ = np.arange(y_.size)

sample_n = 50
x = np.random.choice(x_, sample_n)
y = [y_[i] for i in x]

param = {'lambda': 5, 'sigma': 2}
result = gp.regression(x, x_, y, param)

K = gp.squared_exponential(x, x, param)
K_ = gp.squared_exponential(x, x_, param).T
K__ = gp.squared_exponential(x_, x_, param)

y_star = np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))), y)
cov = K__ - np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))),K_.T)

plt.figure(2)
plt.plot(x_, y_)
plt.plot(x_, y_star)
