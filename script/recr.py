# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:21:55 2016

@author: bagas
"""

from thesis import config
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

style.use('thesis')

recr = np.load(config.ts + 'recr.npy')
recr_res = np.load(config.ts + 'recr_result.npy')

idx = pd.date_range('1950-01', '1987-09', freq='MS')
recr_ts = pd.Series(data=recr.flatten(), index=idx)

recr_1980 = recr_ts['1980':'1987'].values
x = np.arange(len(recr_1980)) + 1
x_ = np.arange(x[-1], x[-1]+len(recr_res[:, 0]))+1

fig, ax = plt.subplots()
ax.plot(x, recr_1980)
ax.plot(x_, recr_res[:, 0])
ax.set_xlabel('time')
ax.set_ylabel('number of fish')
fig.savefig(config.report_image + 'recr.pdf')