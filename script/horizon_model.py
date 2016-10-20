# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:33:08 2016

@author: bagas
"""

from thesis import data
import numpy as np
import pandas as pd

start_date = pd.Timestamp('2016-01-01')
folds = 10

# in minute
fold_length = 60

# training duration in weeks
training_duration = 2

start_stop = []

for i in range(folds):
    start_date = start_date + i * pd.tseries.offsets.Minute(fold_length)
    stop_date = start_date + pd.tseries.offsets.Week(training_duration)
    start_stop.append((str(start_date), str(stop_date)))