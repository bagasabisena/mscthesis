# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:50:56 2016

@author: bagas
"""
from __future__ import division
import gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm

style.use('seaborn-whitegrid')

x = np.linspace(0,10, num=100)
mu = gp.mean_zero(x)
cov = gp.squared_exponential(x,x,{'lambda': 5, 'sigma': 1})
model = {'mu': mu, 'cov': cov}

for i in np.arange(5):
    y = gp.sample(x, model)
    plt.plot(x,y)

#y = gp.fit(x,model)
#plt.plot(x,y, c)
