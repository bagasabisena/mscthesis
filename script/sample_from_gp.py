# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:56:06 2016

@author: bagas
"""

import numpy as np
import pandas as pd
import GPy

import matplotlib.pyplot as plt
from matplotlib import style
style.use('thesis')

from thesis import data
from thesis import gp
from thesis import pipeline
from thesis import evaluation as ev
from thesis import helper
from thesis import config

def prior_sample(x, k, N=5):
    cov = k.K(x)
    mean = np.zeros(cov.shape[0])
    fig, ax = plt.subplots()
    ax.set_xlim([-5, 5])
    for i in range(N):
        Z = np.random.multivariate_normal(mean, cov)
        ax.plot(x, Z.T)
    return fig, ax

def prior_sample2(x, k, ax, N=5):
    cov = k.K(x)
    mean = np.zeros(cov.shape[0])
    for i in range(N):
        Z = np.random.multivariate_normal(mean, cov)
        ax.plot(x, Z.T)
        ax.set_xlim([np.min(x), np.max(x)])
        ax.set_yticklabels([])
x = np.linspace(-5,5.,500).reshape([-1, 1])
k = GPy.kern.RBF(1, lengthscale=1)
ks = [GPy.kern.Linear(1),
      GPy.kern.RBF(1, lengthscale=1),
      GPy.kern.StdPeriodic(1),
      GPy.kern.RatQuad(1, lengthscale=0.1, power=5)]
fig, axes = plt.subplots(4, 1, sharex=True)
for ax, k in zip(axes, ks):
    prior_sample2(x, k, ax, N=3)

fig.tight_layout()
fig.savefig(config.report_image + 'sample2.pdf')
