# -*- coding: utf-8 -*-
"""
Created on Wed May 25 21:25:24 2016

@author: bagas
"""

import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import data


def moving_average(arr, window):
    filtered = pd.rolling_mean(arr, window)
    return filtered


def savitsky_golay(arr, window, poly):
    filtered = savgol_filter(arr, window, poly)
    return filtered


def downsample(data_obj, num_sample):
    assert isinstance(data_obj, data.Data), 'data_obj is not a Data instance'
    # assert (num_sample < 1 and num_sample > 0) or num_sample >= 1, 'it has to be either ratio or positive integer'
    N = data_obj.x_train.shape[0]

    if (num_sample < 1 and num_sample > 0):
        size = int(np.round(num_sample * N))
    elif (num_sample >= 1):
        size = num_sample
    else:
        raise ValueError('it has to be either ratio or positive integer')

    rand_idx = np.random.choice(N, size=size, replace=False)
    rand_idx_sorted = np.sort(rand_idx)
    x = data_obj.x_train[rand_idx_sorted, :]
    y = data_obj.y_train[rand_idx_sorted, :]
    return x, y