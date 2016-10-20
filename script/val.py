# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:47:58 2016

@author: bagas
"""

import pandas as pd
import numpy as np
from thesis import data
from thesis import helper
from thesis import predictor

training_start = '2016-01-20 08:50:00'
training_finish = '2016-01-22 08:50:00'
test_start = pd.Timestamp(training_finish) + pd.tseries.offsets.Minute(5)
test_finish = pd.Timestamp(training_finish) + 19 * pd.tseries.offsets.Minute(5)

conf = helper.read_config()
weather = data.Data.prefetch_data(conf, sample=True)
wind = data.WindData(training_start, training_finish,
                     test_start, test_finish, skformat=True, data=weather)

res = pd.read_csv(conf.get('folder', 'output') + 'runner3_time.csv')

y_true = res.y_true.values.reshape([-1, 1])
print np.hstack([y_true, wind.y_test])

y_pred, cov = predictor.predictor_rbf(wind.x_train, wind.y_train,
                                  wind.x_test, None)

y_pred_res = res.y_pred.values.reshape([-1, 1])
cov_res = res['cov'].values.reshape([-1, 1])

print np.hstack([y_pred_res, y_pred])
print np.hstack([cov_res, cov])