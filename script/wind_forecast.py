# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:15:51 2016

@author: bagas
"""
from __future__ import division
import pandas as pd
import numpy as np
import gp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import optimize

style.use('seaborn-whitegrid')

data_folder = '../data/weather/'

data = pd.read_csv(data_folder + 'Capelle.csv',
                   parse_dates=['DateTime'],
                   index_col=['DateTime'])
                 
#%% wind speed data
wind_speed = data['WindSpd_{Avg}']
y = wind_speed['2016/1/1':'2016/1/7'].astype('float')
x = np.arange(y.size)

y_ = wind_speed['2016/1/8']
x_ = np.arange(y.size, y.size + y_.size)

#def loglikelihood(var, x=x, y=y):
#    param = {'lambda': var[0], 'sigma': var[1]}
#    K = gp.squared_exponential(x,x,param)
#    Kinv = np.linalg.inv(K+10e-6 * np.eye(x.size))
#    N = y.size
#    a = 0.5 * np.log(np.linalg.det(K+10e-6 * np.eye(x.size)))
#    b = 0.5 * np.dot(np.dot(y,Kinv), y)
#    c = (N/2) * np.log(2*np.pi)
#    return a + b + c
#
#res = optimize.minimize(loglikelihood,
#                        x0=[0.5,0.5])
                        
#bounds=[(0, None), (0, None)]

param = {'lambda': 5, 'sigma': 2}

K = gp.squared_exponential(x, x, param)
K_ = gp.squared_exponential(x, x_, param).T
K__ = gp.squared_exponential(x_, x_, param)

y_star = np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))), y)
cov = K__ - np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))),K_.T)

plt.figure(2)
plt.plot(x_, y_)
plt.plot(x_, y_star)
plt.legend(['Actual', 'GP Regression'])


