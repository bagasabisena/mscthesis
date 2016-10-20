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

try:
    import cPickle as pickle
except:
    import pickle


def run_linear_regression(window, horizons, args):
    y_lr = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
        model_lr = linear_model.LinearRegression()
        model_lr.fit(data_obj.x_train, data_obj.y_train)
        y_pred = model_lr.predict(data_obj.x_test)
        y_lr.append(y_pred)

    y_lr = np.vstack(y_lr)
    y_lr = y_lr.flatten()

    return y_lr


def run_bayesian_lr(window, horizons, args):
    y_lr = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
        model_lr = linear_model.BayesianRidge()
        model_lr.fit(data_obj.x_train, data_obj.y_train)
        y_pred = model_lr.predict(data_obj.x_test)
        y_lr.append(y_pred)

    y_lr = np.vstack(y_lr)
    y_lr = y_lr.flatten()

    return y_lr


def run_narx(window, horizons, args):
    y_narx = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
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


def run_rbfiso(window, horizons, args):
    y_narx = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
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


def run_rbfard(window, horizons, args):
    y_narx = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
        kernel = GPy.kern.RBF(data_obj.x_train.shape[1], ARD=True)
        model = GPy.models.GPRegression(data_obj.x_train,
                                        data_obj.y_train,
                                        kernel=kernel)

        model.optimize_restarts(num_restarts=10,
                                robust=True,
                                verbose=True)
        y_pred, cov = model.predict(data_obj.x_test)
        y_narx.append(y_pred) # get scalar from [[y]] array

    y_narx = np.vstack(y_narx)
    y_narx = y_narx.flatten()

    return y_narx


def run_rqiso(window, horizons, args):
    y_narx = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
        kernel = GPy.kern.RatQuad(data_obj.x_train.shape[1])
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


def run_rqard(window, horizons, args):
    y_narx = []
    _, y_train, _, y_test = window
    extra_param = args['extras']
    ar_order = extra_param.get('lag', 5)
    for h in horizons:
        data_obj = data.HorizonWindowData(y_train, y_test, h=h, l=ar_order)
        kernel = GPy.kern.RatQuad(data_obj.x_train.shape[1], ARD=True)
        model = GPy.models.GPRegression(data_obj.x_train,
                                        data_obj.y_train,
                                        kernel=kernel)

        model.optimize_restarts(num_restarts=10,
                                robust=True,
                                verbose=False)
        y_pred, cov = model.predict(data_obj.x_test)
        y_narx.append(y_pred)  # get scalar from [[y]] array

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


def run_persistence(window, horizons, args):
    _, y_train, _, _ = window
    y_pred = y_train[-1]

    # create empty array with the number of horizons
    # and fill it
    y_persistence = np.empty(len(horizons))
    y_persistence.fill(y_pred)
    return y_persistence

def run_auto_arima(window, horizons, args):
    numpy2ri.activate()
    forecast = importr('forecast')
    _, y_train, _, _ = window
    extra_param = args['extras']
    fit = forecast.auto_arima(y_train,
                              stationary=extra_param.get('stationary', False),
                              seasonal=extra_param.get('seasonal', True),
                              stepwise=extra_param.get('stepwise', False))

    order = list(fit[6])
    arima_str = 'ARIMA({0},{5},{1}) ({2},{6},{3})_{4}'
    print arima_str.format(*order)
    result = forecast.forecast(fit, h=max_h, level=np.array([0.95]))
    y_pred = numpy2ri.ri2py(result[3])
    y_arima = y_pred[np.array(horizons) - 1]
    return y_arima


def run_greedy(window, horizons, args):
    extra_param = args['extras']

    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    greedy = search.GreedySearch(data_obj,
                                 base_kernels,
                                 max_kernel=extra_param.get('max_kernel', 4),
                                 max_mult=extra_param.get('max_mult', 2),
                                 obj_fun=extra_param.get('obj_fun', 'bic'))

    preloaded_model = extra_param.get('preloaded_model', None)

    if preloaded_model is not None:
        with open(preloaded_model, 'rb') as f:
            result = pickle.load(f)
    else:
        model, bic, result = greedy.search()

    # model, bic, result = greedy.search()
    model_name = '{}{}_greedy.pkl'.format(config.output,
                                          args['output'])
    result.save(model_name)
    best_result, score = result.best
    model_greedy = best_result.build_model(data_obj.x_train, data_obj.y_train)
    y_pred, cov = model_greedy.predict(data_obj.x_test)
    y_greedy = y_pred.flatten()[np.array(horizons) - 1]
    return y_greedy


def run_greedy_cv(window, horizons, args):
    extra_param = args['extras']

    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)

    y_greedy = []

    for h in horizons:
        greedy = search.CVSearch(data_obj,
                                 base_kernels,
                                 h,
                                 extra_param.get('num_folds', 5),
                                 max_kernel=extra_param.get('max_kernel', 4),
                                 max_mult=extra_param.get('max_mult', 2),
                                 fixed=extra_param.get('fixed', True),
                                 eval_fun=extra_param.get('eval_fun', 'mae'))

        preloaded_model = extra_param.get('preloaded_model', None)

        if preloaded_model is not None:
            with open(preloaded_model, 'rb') as f:
                result = pickle.load(f)
        else:
            model, bic, result = greedy.search()

        # model, bic, result = greedy.search()
        model_name = '{}{}_{}_greedy.pkl'.format(config.output,
                                                 args['output'], h)
        result.save(model_name)
        best_result, score = result.best
        best_result.build_kernel()
        best_result.optimise()
        model_greedy = GPy.models.GPRegression(data_obj.x_train,
                                               data_obj.y_train,
                                               kernel=best_result.kernel,
                                               noise_var=best_result.param_noise)
        # model_greedy = best_result.build_model(data_obj.x_train, data_obj.y_train)
        model_greedy.optimize_restarts(robust=True, verbose=False)
        y_pred, cov = model_greedy.predict(data_obj.x_test)
        y_greedy.append(y_pred[h-1])

    y_greedy = np.vstack(y_greedy)
    y_greedy = y_greedy.flatten()

    return y_greedy


def run_exhaustive(window, horizons, args):
    extra_param = args['extras']
    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    exhaustive = search.ExhaustiveSearch(data_obj,
                                         base_kernels,
                                         max_kernel=extra_param.get('max_kernel', 4),
                                         max_mult=extra_param.get('max_mult', 2),
                                         obj_fun=extra_param.get('obj_fun', 'bic'))

    print exhaustive.num_combination

    model, bic, result = exhaustive.search()
    model_name = '{}{}_ex.pkl'.format(config.output,
                                      args['output'])
    result.save(model_name)
    best_result, score = result.best
    model_exhaustive = best_result.build_model(data_obj.x_train,
                                               data_obj.y_train)
    y_pred, cov = model_exhaustive.predict(data_obj.x_test)
    y_exhaustive = y_pred.flatten()[np.array(horizons) - 1]
    return y_exhaustive


def run_exhaustive_cv(window, horizons, args):
    extra_param = args['extras']

    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)

    y_exhaustive = []

    for h in horizons:
        greedy = search.CVExhautiveSearch(data_obj,
                                          base_kernels,
                                          h,
                                          extra_param.get('num_folds', 5),
                                          max_kernel=extra_param.get('max_kernel', 4),
                                          max_mult=extra_param.get('max_mult', 2),
                                          fixed=extra_param.get('fixed', True),
                                          eval_fun=extra_param.get('eval_fun', 'mae'))

        preloaded_model = extra_param.get('preloaded_model', None)

        if preloaded_model is not None:
            with open(preloaded_model, 'rb') as f:
                result = pickle.load(f)
        else:
            model, bic, result = greedy.search()

        # model, bic, result = greedy.search()
        model_name = '{}{}_{}_ex.pkl'.format(config.output,
                                                 args['output'], h)
        result.save(model_name)
        best_result, score = result.best
        best_result.build_kernel()
        best_result.optimise()
        model_exhaustive = GPy.models.GPRegression(data_obj.x_train,
                                                   data_obj.y_train,
                                                   kernel=best_result.kernel,
                                                   noise_var=best_result.param_noise)
        # model_exhaustive = best_result.build_model(data_obj.x_train, data_obj.y_train)
        model_exhaustive.optimize_restarts(robust=True, verbose=False)
        y_pred, cov = model_exhaustive.predict(data_obj.x_test)
        y_exhaustive.append(y_pred[h-1])

    y_exhaustive = np.vstack(y_exhaustive)
    y_exhaustive = y_exhaustive.flatten()

    return y_exhaustive


def run_true_kernel(window, horizons, args):
    extra_param = args['extras']
    x_train, y_train, x_test, y_test = window

    x_train = x_train.reshape([-1, 1])
    y_train = y_train.reshape([-1, 1])
    x_test = x_test.reshape([-1, 1])
    y_test = y_test.reshape([-1, 1])

    model_id = extra_param.get('model', 2)

    if model_id == 2:
        kernel = GPy.kern.StdPeriodic(1) + GPy.kern.RBF(1)
    elif model_id == 4:
        kernel1 = GPy.kern.Linear(1) + GPy.kern.RBF(1)
        kernel2 = GPy.kern.StdPeriodic(1) * GPy.kern.Linear(1)
        kernel = kernel1 + kernel2
    else:
        raise NotImplementedError('not implemented')

    model = GPy.models.GPRegression(x_train, y_train, kernel=kernel)
    np.save(config.output + args['output'] + 'true', model.param_array)
    y_pred, cov = model.predict(x_test)
    y_truekernel = y_pred.flatten()[np.array(horizons) - 1]
    return y_truekernel


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
    return vars(parser.parse_args(sys_argv))


def run(args):
    global max_h
    # -------- parse the argument ---------------
    syn_data = np.load(config.ts + args['input'] + '.npz')
    x = syn_data['x']
    y = syn_data['y_']
    windows = data.create_window(x, y, args.get('max_h', 20),
                                 args.get('num_window', 10),
                                 training_ratio=args.get('training_ratio', 1.0),
                                 fold_gap=args.get('fold_gap', 1))
    horizons = args.get('horizons', [5])
    max_h = max(horizons)

    model_names = args['names']
    predictor_functions = [globals()[p] for p in args['predictors']]

    output_name = args['output']

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
