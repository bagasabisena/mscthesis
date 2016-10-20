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

from thesis import gp as gp2

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

sample_n = 144
x = np.random.choice(x_, sample_n)
y = np.array([y_[i] for i in x])

param = {'lambda': 5, 'sigma': 2}

gp_regressor = gp2.GaussianProcess(param=param, correction=10e-2)
gp_regressor.fit(x, y)
y_pred = gp_regressor.predict(x_)[0]

plt.figure(1)
plt.plot(x_, y_)
plt.plot(x_, y_pred)
plt.legend(['Actual', 'GP Regression'])