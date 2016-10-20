# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:01:14 2016

@author: bagas
"""

from thesis import gp
from thesis import pipeline
from thesis import evaluation as ev
from thesis import data
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-whitegrid')

p = pipeline.Pipeline(
        data=data.SinDataForecast(skformat=True),
        transformer=None,
        predictor=gp.GaussianProcessGPy(),
        evaluator=[ev.mean_square_error]
    )

p.start()
print p.error

plt.figure(1)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])

#%%
p = pipeline.Pipeline(
        data=data.WindData('2016/1/1', '2016/1/7', '2016/1/8', '2016/1/8', None, None, skformat=True),
        transformer=None,
        predictor=gp.GaussianProcessGPy(),
        evaluator=[ev.mean_square_error]
    )

p.start()

plt.figure(2)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])

#%%

p = pipeline.Pipeline(
        data=data.WindDataRandom('2016/1/1', skformat=True),
        transformer=None,
        predictor=gp.GaussianProcessGPy(),
        evaluator=[ev.mean_square_error]
    )

p.start()

plt.figure(3)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])