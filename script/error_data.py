# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:34:45 2016

@author: bagas
"""

from thesis import helper
from thesis import data

import pandas as pd

raw_data = data.Data.prefetch_data2()

folds = helper.load_fold('random5.date')
folds = map(
    lambda x: helper.generate_training_date(x, 2, pd.offsets.Day()),
    folds
)

start, stop = folds[2]
output_feature = 'WindSpd_{Avg}'
input_feature = ['WindSpd_{Avg}',
                 'WindDir_{Avg}',
                 'Tair_{Avg}',
                 'RH_{Avg}']
l = 5
time_as_feature = False
scale_x = True
scale_y = False

data_obj = data.HorizonData(start, stop, 1, output_feature,
                            input_feature=input_feature,
                            l=5,
                            time_as_feature=time_as_feature,
                            data=raw_data,
                            scale_x=True)
