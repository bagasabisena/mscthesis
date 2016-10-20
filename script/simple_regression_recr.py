# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:58:38 2016

@author: bagas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:29:17 2016

@author: bagas
"""

#%%

from thesis import config
from thesis import data
from sklearn import linear_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('thesis')

recr = np.load(config.ts + 'recr.npy')
x = np.arange(recr.shape[0]).reshape([-1,1])
x_ = np.arange(453, 453+24).reshape([-1, 1])

linreg = linear_model.LinearRegression()
linreg.fit(x, recr)
y_ = linreg.predict(x_)

idx = pd.date_range('1950-01', '1987-09', freq='MS')
recr_ts = pd.Series(data=recr.flatten(), index=idx)

recr_1980 = recr_ts['1980':'1987'].values
x = np.arange(len(recr_1980)) + 1
x_ = np.arange(x[-1], x[-1]+len(y_[:, 0]))+1

#%%

fig, ax = plt.subplots()
ax.plot(x, recr_1980)
ax.plot(x_, y_[:, 0])
ax.set_xlabel('time')
ax.set_ylabel('number of fish')
fig.savefig(config.report_image + 'recr_linreg.pdf')