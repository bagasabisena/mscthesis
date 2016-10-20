# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:09:01 2016

@author: bagas
"""
from thesis import data
from thesis import evaluation as ev
from thesis import helper
import pandas as pd
import gc
import numpy as np
from sklearn import linear_model
import sys


def to_df(results, start_stop, horizons):
    iterables = [map(lambda x: x[0], start_stop), horizons]
    midx = pd.MultiIndex.from_product(iterables)
    result_df = pd.DataFrame(results, index=midx, columns=['y_true', 'y_pred', 'mse'])
    return result_df


def predictor(x_train, y_train, x_test):
    X = x_train
    y = y_train
    X_ = x_test
    # y_ = p.y_test

    model = linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X_)
    return y_pred


def training(h, start, finish, output_feature, input_feature, weather_data, l=1, time_as_feature=False):
    data_class = data.HorizonData(start, finish, h, output_feature, input_feature=input_feature, l=l,
                             time_as_feature=time_as_feature,
                             data=weather_data, skformat=True)

    # pipe = pipeline.Pipeline3(data=model, predictor=predictor,
    #                           evaluator=[ev.mean_square_error, ev.mean_absolute_error, ev.r2_score],
    #                           output=None)

    y_ = predictor(data_class.x_train, data_class.y_train, data_class.x_test)

    mse = ev.mean_square_error(data_class.y_test, y_)
    y_true = data_class.y_test[0, 0]

    # x_train = np.linspace(-2, 2).reshape([-1, 1])
    # y_train = np.sin(x_train).reshape([-1, 1])
    # x_test = np.linspace(2, 3).reshape([-1, 1])
    # y_test = np.sin(x_test)
    # y_, cov = predictor(x_train, y_train, x_test)
    # mse = ev.mean_square_error(y_test, y_)
    # y_true = y_test[0, 0]

    y_pred = y_[0, 0]
    return [y_true, y_pred, mse]
    # return (1, 2, 3, 4)


# def run_test(horizons, start, finish, output_feature, input_feature, l=1, time_as_feature=False):
#     results = [training(h, start, finish, output_feature, input_feature, l, time_as_feature) for h in horizons]
#     return results


def load_data(conf):
    return data.WindData.prefetch_data(conf)

# @profile
# def run_fold(horizons, output_feature, input_feature, l=1, time_as_feature=False):
#     results = []
#     for ss in start_stop:
#         r = run_test(horizons, ss[0], ss[1], output_feature, input_feature, l=l, time_as_feature=time_as_feature)
#         results.append(r)
#         print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#     result_df = to_df(list(chain.from_iterable(results)), start_stop, horizons)
#     result_df.to_csv(output_path)


def create_fold(start_date, folds, fold_length, training_duration):
    start_stop = []
    for i in range(folds):
        start_date = start_date + pd.tseries.offsets.Minute(fold_length)
        stop_date = start_date + pd.tseries.offsets.Day(training_duration)
        start_stop.append((str(start_date), str(stop_date)))
    return start_stop


def run(l):
    # --------configuration-------------
    name = "mlag_linreg"
    start_date = pd.Timestamp('2016-01-01')
    folds = 10
    fold_length = 60  # in minute
    training_duration = 14  # in day
    horizons = [1, 6, 12, 48, 96, 144, 288, 576]
    # horizons = [1, 6, 12, 48]
    # horizons = range(1, 11)
    output_feature = 'WindSpd_{Avg}'
    reported_feature = 3
    input_feature = input_feature = ['Tair_{Avg}', 'RH_{Avg}', 'e_{Avg}', 'WindSpd_{Max}',
                  'WindSpd_{Std}', 'WindSpd_{Avg}', 'WindDir_{Avg}', 'WindDir_{Std}']
    l = l
    time_as_feature = False
    conf = helper.read_config()
    weather_data = load_data(conf)
    # --------configuration-------------

    fold_detail = create_fold(start_date, folds, fold_length, training_duration)
    output_dir = conf.get('folder', 'output')
    output_path = '%s%s.csv' % (output_dir, name)

    global final_result
    final_result = np.empty([len(horizons) * folds, reported_feature])

    i = 0
    for f, fold in enumerate(fold_detail):
        for h in horizons:
            final_result[i, :] = training(h, fold[0], fold[1], output_feature,
                                          input_feature, weather_data,
                                          l, time_as_feature)
            i = i+1
            gc.collect()

            # final_result[i, :] = [1, 2, 3, 4]

    result_df = to_df(final_result, fold_detail, horizons)
    result_df.to_csv(output_path)

    helper.send_email('bagasabisena@gmail.com',
                  'job_notification',
                  'job %s finished. Output %s generated' % (name, output_path))

if __name__ == '__main__':
    ls = [1, 3, 5, 10]
    for l in ls:
        run(l)
