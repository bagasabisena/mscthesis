# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:23:18 2016

@author: bagas
"""

from thesis import data, config, lag
import numpy as np
import matplotlib.pyplot as plt
import GPy

syn_data = np.load(config.ts + 'osynthetic_2_0_1.npz')
x = syn_data['x']
y = syn_data['y_']

windows = data.create_window(x, y, 10, 20)
window = windows[-1]

lags = np.arange(30)+1
data_params = dict(window=window)
model_params = {'kernel': GPy.kern.RBF,
                'ard': False}
eval_params = dict(num_fold=5)
lag_cv = lag.LagCV(lags, 'cv_mae', 'synth', 'gp', 10, data_params,
                   model_params, eval_params)
lag_cv.run(verbose=True)

plt.plot(lag_cv.lags, lag_cv.scores)
