# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:30:11 2016

@author: bagas
"""

from thesis import data
from thesis import gp
from thesis import evaluation as ev
from thesis import pipeline
import numpy as np
from sklearn import cross_validation as cv

wind_data = data.WindData('2016/1/1',
                          '2016/1/7',
                          '2016/1/8 01:00:00',
                          '2016/1/8 02:00:00',
                          '2016/1/8 00:00:00',
                          '2016/1/8 01:00:00', data=weather_data)


l = np.array([0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
sigma = np.array([0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

result = []
for i in xrange(l.size):
    for j in xrange(sigma.size):
        param={'lambda': l[i], 'sigma': sigma[j]}
        gpr = gp.GaussianProcess(param=param, correction=1,optimize=False)
        gpr.fit(wind_data.x_train, wind_data.y_train)
        y_pred, _ = gpr.predict(wind_data.x_val)
        score = ev.mean_square_error(wind_data.y_val, y_pred)
        result.append([l[i], sigma[j], score])


#%%
from matplotlib import pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

resnp = np.array(result)
min_i = np.argmin(resnp[:,2])
p = pipeline.Pipeline(data=wind_data,
                      transformer=None,
                      predictor=gp.GaussianProcess(param={'lambda': resnp[min_i,0], 'sigma': resnp[min_i,1]},
                                                   optimize=False),
                      evaluator=[ev.mean_square_error])

p.start()

plt.figure(1)
plt.plot(p.data.x_test, p.y_pred[0])
plt.plot(p.data.x_test, p.data.y_test)
plt.legend(['Predicted GP regression', 'Actual'])

