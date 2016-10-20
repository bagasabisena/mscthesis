# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:09:01 2016

@author: bagas
"""
from thesis import data
from thesis import pipeline
from thesis import evaluation as ev
from thesis import helper
import GPy
import pandas as pd


def to_df(result):
    result_df = pd.DataFrame(result, columns=['h', 'y_true', 'y_pred', 'mse', 'mae', 'r2', 'cov'])
    result_df.set_index('h', inplace=True)
    return result_df


def predictor(p):
    X = p.x_train
    y = p.y_train
    X_ = p.x_test
    # y_ = p.y_test

    rbf = GPy.kern.RBF(X.shape[1])
    model = GPy.models.GPRegression(X, y, kernel=rbf)
    model.optimize(messages=False, max_f_eval=100)
    y_pred, cov = model.predict(X_)
    return y_pred, cov, model


def run_test(horizons, start, finish, output_feature, input_feature, l=1, time_as_feature=False):
    result = []

    for h in horizons:
        model = data.HorizonData(start, finish, h, output_feature, input_feature=input_feature, l=l,
                                 time_as_feature=time_as_feature,
                                 data=weather_data, skformat=True)

        pipe = pipeline.Pipeline3(data=model, predictor=predictor,
                                  evaluator=[ev.mean_square_error, ev.mean_absolute_error, ev.r2_score],
                                  output=None)

        pipe.start()
        y_true = pipe.y_test[0, 0]
        y_pred = pipe.y_[0, 0]
        mse = pipe.error[0][1]
        mae = pipe.error[1][1]
        r2 = pipe.error[2][1]
        cov = pipe.cov[0, 0]
        result.append((h, y_true, y_pred, mse, mae, r2, cov))

    result_df = to_df(result)
    return result_df

# ---------------------
name = "time_h24_576"
start = '2016-01-01'
finish = '2016-01-07'
h_hour = [i*24 for i in range(1, 11)]
h_day = [i*288 for i in range(1, 3)]
horizons = h_hour + h_day
output_feature = 'WindSpd_{Avg}'
# ----------------------

conf = helper.read_config()
weather_data = data.WindData.prefetch_data(conf)
output_dir = conf.get('folder', 'output')
output_path = '%s%s.csv' % (output_dir, name)

result_df = run_test(horizons, start, finish, output_feature, None, l=1, time_as_feature=False)
result_df.to_csv(output_path)

helper.send_email('bagasabisena@gmail.com',
                  'job_notification',
                  'job %s finished. Output %s generated' % (name, output_path))
