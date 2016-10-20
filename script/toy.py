# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:14:12 2016

@author: bagas
"""
from __future__ import division
import numpy as np
import gp
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-whitegrid')

#%%
x = np.array([-1.5, -1, -0.75, -0.40, -0.25, 0.0])
y = np.array([-1.7, -1.2, -0.3, 0.2, 0.5, 0.8])
x_ = np.array([0.2])

#%%

# estimate for very large number
param = {'lambda': 1, 'sigma': 1}
x_ = np.linspace(-2, 1, num=100)
K = gp.squared_exponential(x,x,param)
K_ = gp.squared_exponential(x,x_, param).T
K__ = gp.squared_exponential(x_,x_, param)

y_ = np.dot(np.dot(K_, np.linalg.inv(K)), y)
cov = K__ - np.dot(np.dot(K_, np.linalg.inv(K)),K_.T)

plt.ylim([-2.5, 2.5])
plt.plot(x_,y_)
plt.plot(x, y, '.')
