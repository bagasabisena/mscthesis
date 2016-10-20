# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:15:14 2016

@author: bagas
"""

import pandas as pd

a = pd.read_csv('output/test_time.csv', index_col=[0, 1])
a.index.names = [None, None]
b = pd.read_csv('output/test_lag_1.csv', index_col=[0, 1])
b.index.names = [None, None]