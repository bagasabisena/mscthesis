# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:28:49 2016

@author: bagas
"""
from __future__ import division
import numpy as np
from scipy import optimize
import gp
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-whitegrid')

np.random.seed(0)
n = 1000

#param = {'lambda': 2.7932, 'sigma': 0.6187}
param = {'lambda': 1, 'sigma': 1}

sample_n = 100

x_ = np.linspace(-10,10,n)
x = np.random.choice(x_, sample_n)
#y = 1 + 0.05 * x + np.sin(x) / x + 0.2 * np.random.randn(n)
y = 1 + 0.05 * x + np.sin(x) / x + 0.2 * np.random.randn(sample_n)
y_= 1 + 0.05 * x_ + np.sin(x_) / x_ + 0.2 * np.random.randn(n)
y_true = 1 + 0.05 * x_ + np.sin(x_) / x_

def loglikelihood(var, x=x, y=y):
    param = {'lambda': var[0], 'sigma': var[1]}
    K = gp.squared_exponential(x,x,param)
    Kinv = np.linalg.inv(K+10e-6 * np.eye(x.size))
    N = y.size
    a = 0.5 * np.log(np.linalg.det(K+10e-6 * np.eye(x.size)))
    b = 0.5 * np.dot(np.dot(y,Kinv), y)
    c = (N/2) * np.log(2*np.pi)
    return a + b + c

#res = optimize.minimize(loglikelihood, x0=[0.5,0.5])


K = gp.squared_exponential(x, x, param)
K_ = gp.squared_exponential(x, x_, param).T
K__ = gp.squared_exponential(x_, x_, param)

y_star = np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))), y)
cov = K__ - np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))),K_.T)

plt.figure(1)
#plt.plot(x,y,'.r')
plt.plot(x_,y_, '.', ms=0.5)
plt.plot(x_, y_star, linewidth=1)
plt.plot(x_, y_true, linewidth=1)
plt.legend(('Function(with noise)', 'GP regression', 'Actual function (noiseless)'))

def linear(a,b):
    return np.outer(a,b)

K = linear(x, x)
K_ = linear(x, x_).T
K__ = linear(x_, x_)

y_star = 1.2* np.ones(x_.size) + np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))), (y-1.2*np.ones(x.size)))
cov = K__ - np.dot(np.dot(K_, np.linalg.inv(K+10e-6 * np.eye(x.size))),K_.T)

plt.figure(2)
#plt.plot(x,y,'.r')
plt.plot(x_,y_, '.', ms=0.5)
plt.plot(x_, y_star, linewidth=1)
plt.plot(x_, y_true, linewidth=1)
plt.legend(('', 'GP regression', 'Actual function (noiseless)'))

#result = gp.regression(x,x_,y, param)
#y_star = result['mu']
#
#plt.figure(1)
#plt.ylim([0,3])
#plt.plot(x_,y_)
#
#plt.figure(2)
#plt.ylim([0,3])
#plt.plot(x_,y_)
#plt.plot(x_,y_star)
#plt.plot(x,y,'.')