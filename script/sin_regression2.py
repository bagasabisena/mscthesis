# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:54:20 2016

@author: bagas
"""

from __future__ import division
import numpy as np
from scipy import optimize
import gp
import matplotlib.pyplot as plt
from matplotlib import style
from thesis import gp as gp2

style.use('seaborn-whitegrid')

np.random.seed(0)
n = 100

#param = {'lambda': 2.7932, 'sigma': 0.6187}
param = {'lambda': 1, 'sigma': 1}

sample_n = 50

x_ = np.linspace(-10,10,n)
x = np.random.choice(x_, sample_n)
#y = 1 + 0.05 * x + np.sin(x) / x + 0.2 * np.random.randn(n)
y = 1 + 0.05 * x + np.sin(x) / x + 0.2 * np.random.randn(sample_n)
y_= 1 + 0.05 * x_ + np.sin(x_) / x_ + 0.2 * np.random.randn(n)
y_true = 1 + 0.05 * x_ + np.sin(x_) / x_

gp_regressor = gp2.GaussianProcess(correction=0.2)
gp_regressor.fit(x, y)
y_pred, _ = gp_regressor.predict(x_)

plt.figure(1)
#plt.plot(x,y,'.r')
plt.plot(x_,y_, '.', ms=0.5)
plt.plot(x_, y_pred, linewidth=1)
plt.plot(x_, y_true, linewidth=1)
plt.legend(('Function(with noise)', 'GP regression', 'Actual function (noiseless)'))