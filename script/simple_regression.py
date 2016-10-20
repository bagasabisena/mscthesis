# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:29:17 2016

@author: bagas
"""

from thesis import config
from thesis import data
from sklearn import linear_model

import matplotlib.pyplot as plt
from matplotlib import style
style.use('thesis')

co2 = data.TSData('mauna')

linreg = linear_model.LinearRegression()
linreg.fit(co2.x_train, co2.y_train)
y_ = linreg.predict(co2.x_test)

fig, ax = plt.subplots()
ax.plot(co2.x_train, co2.y_train)
ax.plot(co2.x_test, y_)
ax.legend(['observations', 'forecasts'], loc=2)
ax.set_xlabel('time (year)')
ax.set_ylabel('co2 concentration (ppm)')
fig.savefig(config.report_image + 'simple.pdf')