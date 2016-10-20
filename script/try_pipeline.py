# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:11:27 2016

@author: bagas
"""

from thesis import data
from thesis import pipeline
from thesis import gp
from thesis import evaluation
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#%%

p = pipeline.Pipeline(
    data=data.WindData('2016/1/1', '2016/1/7', '2016/1/8', None),
    transformer=[MinMaxScaler()],
    predictor=gp.GaussianProcess(correction=10e-1),
    evaluator=[evaluation.mean_square_error,
               evaluation.mean_absolute_error,
               evaluation.median_absolute_error]
)

p.start()

#%%

style.use('seaborn-whitegrid')
plt.figure(1)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])

#%% sin data
p = pipeline.Pipeline(
    data=data.SinData(),
    transformer=None,
    predictor=gp.GaussianProcess(correction=0.2),
    evaluator=[evaluation.mean_square_error,
               evaluation.mean_absolute_error,
               evaluation.median_absolute_error])

p.start()

plt.figure(2)
plt.plot(p.data.x_test, p.data.y_test, '.', ms=0.5)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_true)
plt.legend(('Function(with noise)', 'Predicted GP regression', 'Actual function (noiseless)'))

#%% wind regression
reload(data)
p = pipeline.Pipeline(
    data=data.WindDataRandom('2016/1/1'),
    transformer=None,
    predictor=gp.GaussianProcess(correction=10e-4),
    evaluator=[evaluation.mean_square_error,
               evaluation.mean_absolute_error,
               evaluation.median_absolute_error])

p.start()

plt.figure(3)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(('Predicted GP regression', 'Actual'))

#%%
