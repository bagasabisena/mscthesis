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


def to_df(results, start_stop, horizons):
    iterables = [map(lambda x: x[0], start_stop), horizons]
    midx = pd.MultiIndex.from_product(iterables)
    result_df = pd.DataFrame(results, index=midx, columns=['y_true', 'y_pred', 'mse', 'mae', 'r2', 'cov'])
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
        result.append((y_true, y_pred, mse, mae, r2, cov))

    # result_df = to_df(result)
    return result

# ---------------------
name = "lag_cv"
start_date = pd.Timestamp('2016-01-01')
folds = 10
fold_length = 60  # in minute
training_duration = 7  # in day
horizons = [1, 6, 12, 48, 96, 144, 288, 576]
output_feature = 'WindSpd_{Avg}'
lags = [1, 3, 5, 10]
# ----------------------

start_stop = []
for i in range(folds):
    start_date = start_date + i * pd.tseries.offsets.Minute(fold_length)
    stop_date = start_date + pd.tseries.offsets.Day(training_duration)
    start_stop.append((str(start_date), str(stop_date)))

conf = helper.read_config()
weather_data = data.WindData.prefetch_data(conf)
output_dir = conf.get('folder', 'output')

for l in lags:
    output_path = '%s%s_l_%d.csv' % (output_dir, name, l)
    result_df = run_test(horizons, start, finish, output_feature, output_feature, l=l, time_as_feature=False)
    result_df.to_csv(output_path)

helper.send_email('bagasabisena@gmail.com',
                  'job_notification',
                  'job %s finished. Output %s generated' % (name, output_path))
