# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:38:13 2016

@author: bagas
"""

from thesis import config
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np

style.use('thesis')

eggs = np.load(config.ts + 'eggs.npy')
ausbeer = np.load(config.ts + 'ausbeer.npy')

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(eggs)
ax1.set_xlabel('time')
ax1.set_ylabel('US dollar')

ax2.plot(ausbeer)
ax2.set_xlabel('time')
ax2.set_xlim([0, 220])
ax2.set_xticks(np.arange(20, 221, 20))
ax2.set_ylabel('megaliters')

fig.tight_layout()
fig.savefig(config.report_image + 'trend_season.pdf')