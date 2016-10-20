# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:30:37 2016

@author: bagas
"""

from thesis import data
from thesis import gp
from thesis import evaluation as ev
from thesis import pipeline
import numpy as np
from sklearn import cross_validation as cv

l = np.array([0.01, 0.1, 1, 2, 5, 10, 20, 50, 100])
sigma = np.array([0.01, 0.1, 1, 2, 5, 10, 20, 50, 100])

sin_data = data.SinDataForecast()

result = []
for i in xrange(l.size):
    for j in xrange(sigma.size):
        param={'lambda': l[i], 'sigma': sigma[j]}
        gpr = gp.GaussianProcess(param=param,correction=10e-6,optimize=False)
        gpr.fit(sin_data.x_train, sin_data.y_train)
        y_pred, _ = gpr.predict(sin_data.x_val)
        score = ev.mean_square_error(sin_data.y_val, y_pred)
        result.append([l[i], sigma[j], score])


#%%
from matplotlib import pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

resnp = np.array(result)
min_i = np.argmin(resnp[:,2])
p = pipeline.Pipeline(data=sin_data,
                      transformer=None,
                      predictor=gp.GaussianProcess(param={'lambda': resnp[min_i,0], 'sigma': resnp[min_i,1]},
                                                   optimize=False),
                      evaluator=[ev.mean_square_error])

p.start()

plt.figure(1)
plt.plot(np.concatenate((p.data.x_train,
                         p.data.x_val,
                         p.data.x_test)),
                      (p.data.y_train,
                       p.data.y_val,
                       p.y_pred[0]))

plt.plot(np.concatenate((p.data.x_train,
                         p.data.x_val,
                         p.data.x_test)),
                      (p.data.y_train,
                       p.data.y_val,
                       p.data.y_test))
plt.legend(['Predicted GP regression', 'Actual'])

