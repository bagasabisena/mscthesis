import GPy
import numpy as np
from sklearn import linear_model
from statsmodels import api as sm

from thesis import config
from thesis import data
from thesis import search

import readline
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

import collections


def run_linear_regression(window, horizons):
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


def run_narx(window, horizons):
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



def run_arima(window, horizons):
    _, y_train, _, _ = window
    order = (5, 0)
    model_arima = sm.tsa.ARMA(y_train, order).fit()
    y_pred, _, _ = model_arima.forecast(max_h)
    y_arima = y_pred[np.array(horizons) - 1]
    return y_arima


def run_auto_arima(window, horizons):
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


def run_greedy(window, horizons):
    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    greedy = search.GreedySearch(data_obj,
                                 base_kernels,
                                 max_kernel=4,
                                 max_mult=2)

    model, bic, result = greedy.search()
    best_result, score = result.best
    model_greedy = best_result.build_model(data_obj.x_train, data_obj.y_train)
    y_pred, cov = model_greedy.predict(data_obj.x_test)
    y_greedy = y_pred.flatten()[np.array(horizons) - 1]
    return y_greedy


def run():
    global max_h
    # -------- edit this ---------------
    # load noise level 0.1
    syn_data = np.load(config.ts + 'synthetic_1.npz')
    x = syn_data['x']
    y = syn_data['y_']
    # maximum horizon=20, number of windows=10
    windows = data.create_window(x, y, 20, 10)
    horizons = [5, 10, 20]
    max_h = max(horizons)

    model_names = ['arima']
    predictor_functions = [run_auto_arima]

    output_name = 'result_synth_sigma1_arima'

    # -------- edit this ---------------

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
            y_pred = fun(window, horizons)
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
    run()
