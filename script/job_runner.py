# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:09:01 2016

@author: bagas
"""
from thesis import data
from thesis import pipeline
from thesis import evaluation as ev
from thesis import helper
from thesis import predictor
import GPy
import pandas as pd
import resource
import gc
from itertools import chain
import numpy as np
import json
import sys


def to_df(results, start_stop, horizons):
    iterables = [map(lambda x: x[0], start_stop), horizons]
    midx = pd.MultiIndex.from_product(iterables)
    result_df = pd.DataFrame(results, index=midx,
                             columns=['y_true', 'y_pred', 'mse', 'cov'])
    return result_df


def training(h, start, finish, output_feature,
             input_feature, weather_data, predictor_fun_name, job,
             l=1, time_as_feature=False, scale_x=False):
    data_obj = data.HorizonData(start, finish, h, output_feature,
                                input_feature=input_feature, l=l,
                                time_as_feature=time_as_feature,
                                data=weather_data, skformat=True,
                                scale_x=False)

    try:
        predictor_fun = getattr(predictor, predictor_fun_name)
    except AttributeError:
        print "no predictor function by the name of %s" % predictor_fun_name
        exit()

    y_, cov = predictor_fun(data_obj.x_train, data_obj.y_train,
                            data_obj.x_test, job=job)

    # force result to be float64
    y_ = y_.astype('float')
    y_test = data_obj.y_test.astype('float')
    mse = ev.mean_square_error(y_test, y_)

    y_true = y_test[0, 0]

    # x_train = np.linspace(-2, 2).reshape([-1, 1])
    # y_train = np.sin(x_train).reshape([-1, 1])
    # x_test = np.linspace(2, 3).reshape([-1, 1])
    # y_test = np.sin(x_test)
    # y_, cov = predictor(x_train, y_train, x_test)
    # mse = ev.mean_square_error(y_test, y_)
    # y_true = y_test[0, 0]

    y_pred = y_[0, 0]
    cov_ = cov[0, 0]
    return [y_true, y_pred, mse, cov_]
    # return (1, 2, 3, 4)


def load_data(conf):
    return data.WindData.prefetch_data(conf)


def create_fold(start_date, folds, fold_length, training_duration):
    start_stop = []
    for i in range(folds):
        start_date = start_date + pd.tseries.offsets.Minute(fold_length)
        stop_date = start_date + pd.tseries.offsets.Day(training_duration)
        start_stop.append((str(start_date), str(stop_date)))
    return start_stop


def generate_start_stop_date(start_date, training_duration,
                             offset=pd.tseries.offsets.Day):
    start_tstamp = pd.Timestamp(start_date)
    finish_tstamp = start_tstamp + offset(training_duration)
    return (str(start_tstamp), str(finish_tstamp))


def random_fold(fold_num, weather_data, save=False):
    start_stop = []
    for i in range(fold_num):
        random_date = str(pd.Timestamp(np.random.choice(weather_data.index)))
        start_stop_date = generate_start_stop_date(random_date, 2)
        start_stop.append(start_stop_date)

    return start_stop


def run(job, weather_data, is_report_email=False):
    # --------configuration-------------
    name = job['name']
    fold_detail = job['folds']
    horizons = job['horizons']
    output_feature = job['output_feature']
    input_feature = job['input_feature']
    l = job['l']
    time_as_feature = job['time_as_feature']
    output_name = job['output_name']
    predictor_fun_name = job['predictor']
    scale_x = job.get('scale_x', False)
    reported_feature = 4
    folds = len(fold_detail)
    # --------configuration-------------

    output_dir = conf.get('folder', 'output')
    output_path = '%s%s.csv' % (output_dir, output_name)

    global final_result
    final_result = np.empty([len(horizons) * folds, reported_feature])

    print name

    i = 0
    for f, fold in enumerate(fold_detail):
        print fold
        for h in horizons:
            print h
            final_result[i, :] = training(h, fold[0], fold[1], output_feature,
                                          input_feature, weather_data,
                                          predictor_fun_name,
                                          l, time_as_feature,
                                          scale_x)
            i = i + 1
            gc.collect()

            # final_result[i, :] = [1, 2, 3, 4]

    result_df = to_df(final_result, fold_detail, horizons)
    result_df.to_csv(output_path)

    if is_report_email:
        helper.send_email('bagasabisena@gmail.com',
                          'job_notification',
                          'job %s finished. Output %s generated' % (
                          name, output_path))


if __name__ == '__main__':
    conf = helper.read_config()
    weather_data = load_data(conf)

    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        job_spec = json.load(f)

    # if folds exist, load it
    fold_name = job_spec.get('folds', None)
    if fold_name is not None:
        folds = helper.load_fold(fold_name)
    else:
        folds = random_fold(10, weather_data, False)

    horizons = helper.parse_horizon(job_spec['horizons'])

    # inject horizon and folds
    job_spec['horizons'] = horizons
    job_spec['folds'] = folds

    for job in job_spec['jobs']:
        # inject horizon and folds
        job['horizons'] = horizons
        job['folds'] = folds
        # inject output name
        job['output_name'] = "%s_%s" % (job_spec['name'], job['name'])
        run(job, weather_data, False)
