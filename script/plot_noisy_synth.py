# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:13:16 2016

@author: bagas
"""

from thesis import config
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('thesis')

files = glob.glob(config.ts + 'osynthetic*')
names = ['2-kernel, SNR=10',
         '2-kernel, SNR=1',
         '4-kernel, SNR=10',
         '4-kernel, SNR=1']

files = [config.ts + 'synthetic_2_0_1.npz',
         config.ts + 'synthetic_2_1.npz',
         config.ts + 'synthetic_1.npz',
         config.ts + 'synthetic_10.npz']

shifts = [2.0, 2.0, 7.0, 7.0]
         
fig, axes = plt.subplots(2, 2, sharey=False, sharex=True)
for name, f, ax, shift in zip(names, files, axes.flatten(), shifts):
    npz_file = np.load(f)
    xlabel = np.arange(npz_file['y_'].shape[0])
    ax.plot(xlabel, npz_file['y_'] + shift)
    ax.set_title(name)
    ax.set_xlim([0, 150])
    
fig.tight_layout()
fig.savefig(config.report_image + 'synth_noisy.pdf')
