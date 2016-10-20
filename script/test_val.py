# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:42:55 2016

@author: bagas
"""

from thesis import data
from thesis import pipeline
from thesis import gp
from thesis import evaluation
from matplotlib import pyplot as plt

#weather_data = data.WindData.prefetch_data()

wind_data = data.WindData('2016/1/1', '2016/1/7', '2016/1/8 01:00:00', '2016/1/8 02:00:00', '2016/1/8 00:00:00', '2016/1/8 01:00:00', data=weather_data, scale_y=True)

p = pipeline.Pipeline(data=wind_data,
                      transformer=None,
                      predictor=gp.GaussianProcess(val=True, correction=1),
                      evaluator=[evaluation.mean_square_error])

p.start()

plt.figure(1)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])

print p.y_pred

#%%
wind_data2 = data.WindData('2016/1/1', '2016/1/7', '2016/1/8 01:00:00', '2016/1/8 02:00:00', None, None, skformat=True)

p2 = pipeline.Pipeline(data=wind_data2,
                      transformer=None,
                      predictor=gp.GaussianProcessGPy(),
                      evaluator=[evaluation.mean_square_error])

p2.start()
plt.figure(2)
plt.plot(p2.data.x_test, p2.y_pred[0])
plt.plot(p2.data.x_test, p2.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])