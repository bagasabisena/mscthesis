import argparse
import json

import GPy
import numpy as np
import sys
from sklearn import linear_model
from statsmodels import api as sm

from thesis import config
from thesis import data
from thesis import search

import readline
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

import collections


def run_linear_regression(window, horizons, args):
    y_lr = []
    _, y_train, _, y_test = window
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=5)
        model_lr = linear_model.LinearRegression()
        model_lr.fit(data_obj.x_train, data_obj.y_train)
        y_pred = model_lr.predict(data_obj.x_test)
        y_lr.append(y_pred)

    y_lr = np.vstack(y_lr)
    y_lr = y_lr.flatten()

    return y_lr


def run_narx(window, horizons, args):
    y_narx = []
    _, y_train, _, y_test = window
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=5)
        kernel = GPy.kern.RBF(data_obj.x_train.shape[1])
        model = GPy.models.GPRegression(data_obj.x_train,
                                        data_obj.y_train,
                                        kernel=kernel)

        model.optimize_restarts(num_restarts=10,
                                robust=True,
                                verbose=False)
        y_pred, cov = model.predict(data_obj.x_test)
        y_narx.append(y_pred) # get scalar from [[y]] array

    y_narx = np.vstack(y_narx)
    y_narx = y_narx.flatten()

    return y_narx



def run_arima(window, horizons, args):
    _, y_train, _, _ = window
    order = (5, 0)
    model_arima = sm.tsa.ARMA(y_train, order).fit()
    y_pred, _, _ = model_arima.forecast(max_h)
    y_arima = y_pred[np.array(horizons) - 1]
    return y_arima


def run_auto_arima(window, horizons, args):
    numpy2ri.activate()
    forecast = importr('forecast')
    _, y_train, _, _ = window
    fit = forecast.auto_arima(y_train,
                              stationary=False,
                              seasonal=True,
                              stepwise=False)

    order = list(fit[6])
    arima_str = 'ARIMA({0},{5},{1}) ({2},{6},{3})_{4}'
    print arima_str.format(*order)
    result = forecast.forecast(fit, h=max_h, level=np.array([0.95]))
    y_pred = numpy2ri.ri2py(result[3])
    y_arima = y_pred[np.array(horizons) - 1]
    return y_arima


def run_greedy(window, horizons, args):
    extra_param = args.extras

    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    greedy = search.GreedySearch(data_obj,
                                 base_kernels,
                                 max_kernel=extra_param.get('max_kernel', 4),
                                 max_mult=extra_param.get('max_mult', 2))

    model, bic, result = greedy.search()
    result.save(args.output + '_result.pkl')
    best_result, score = result.best
    model_greedy = best_result.build_model(data_obj.x_train, data_obj.y_train)
    y_pred, cov = model_greedy.predict(data_obj.x_test)
    y_greedy = y_pred.flatten()[np.array(horizons) - 1]
    return y_greedy


def run_exhaustive(window, horizons, args):
    extra_param = args.extras
    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    exhaustive = search.ExhaustiveSearch(data_obj,
                                         base_kernels,
                                         max_kernel=extra_param.get('max_kernel', 4),
                                         max_mult=extra_param.get('max_mult', 2))

    print exhaustive.num_combination

    model, bic, result = exhaustive.search()
    result.save(args.output + '_result.pkl')
    best_result, score = result.best
    model_exhaustive = best_result.build_model(data_obj.x_train,
                                               data_obj.y_train)
    y_pred, cov = model_exhaustive.predict(data_obj.x_test)
    y_exhaustive = y_pred.flatten()[np.array(horizons) - 1]
    return y_exhaustive


def parse_arguments(sys_argv):
    parser = argparse.ArgumentParser('synthetic data runner')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-n', '--names', nargs='*', required=True)
    parser.add_argument('-p', '--predictors', nargs='*', required=True)
    parser.add_argument('-o', nargs='+', type=int, default=5, dest='horizons')
    parser.add_argument('-m', default=20, type=int, dest='max_h')
    parser.add_argument('-w', default=10, type=int, dest='num_window')
    parser.add_argument('--ratio', default=1.0,
                        type=float, dest='training_ratio')
    parser.add_argument('--gap', default=1, type=int, dest='fold_gap')
    parser.add_argument('-e', '--extra', type=json.loads, dest='extras')
    return parser.parse_args(sys_argv)


def run(args):
    global max_h
    # -------- parse the argument ---------------
    syn_data = np.load(config.ts + args.input + '.npz')
    x = syn_data['x']
    y = syn_data['y_']
    windows = data.create_window(x, y, args.max_h, args.num_window,
                                 training_ratio=args.training_ratio,
                                 fold_gap=args.fold_gap)
    horizons = args.horizons
    max_h = max(horizons)

    model_names = args.names
    predictor_functions = [globals()[p] for p in args.predictors]
    print predictor_functions

    output_name = args.output

    # -------- parse the argument ---------------

    specs = collections.OrderedDict()
    for name, fun in zip(model_names, predictor_functions):
        specs[name] = {
            'fun': fun,
            'result': []
        }

    y_true = []
    for window in windows:
        for spec in specs.values():
            fun = spec['fun']
            y_pred = fun(window, horizons, args)
            spec['result'].append(y_pred)

        y_test = window[3]
        y_true.append(y_test[np.array(horizons) - 1])

    # iterate specs dictionary to create vstack from array of result
    end_results = dict()
    # inject true y to the end_results
    end_results['true'] = np.vstack(y_true)

    for name, spec in specs.items():
        result_stacked = np.vstack(spec['result'])
        end_results[name] = result_stacked

    np.savez(config.output + output_name, **end_results)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print args
    run(args)
