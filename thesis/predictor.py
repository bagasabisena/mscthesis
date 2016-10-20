import GPy
import numpy as np
from sklearn import linear_model
from thesis import search
import copy
import config
try:
    import cPickle as pickle
except:
    import pickle


def predictor_rbf(x_train, y_train, x_test, job):
    X = x_train
    y = y_train
    X_ = x_test
    # y_ = p.y_test

    rbf = GPy.kern.RBF(X.shape[1])
    model = GPy.models.GPRegression(X, y, kernel=rbf)
    model.optimize(messages=False, max_f_eval=100)
    y_pred, cov = model.predict(X_)
    return y_pred, cov


def predictor_rq(x_train, y_train, x_test, job):
    X = x_train
    y = y_train
    X_ = x_test
    # y_ = p.y_test

    k = GPy.kern.RatQuad(X.shape[1])
    model = GPy.models.GPRegression(X, y, kernel=k)
    model.optimize(messages=False)
    y_pred, cov = model.predict(X_)
    return y_pred, cov


def predictor_rbf_ard(x_train, y_train, x_test, job):
    rbf = GPy.kern.RBF(x_train.shape[1], ARD=True)
    model = GPy.models.GPRegression(x_train, y_train, kernel=rbf)
    model.optimize(messages=False)
    y_pred, cov = model.predict(x_test)
    return y_pred, cov


def predictor_rq_ard(x_train, y_train, x_test, job):
    k = GPy.kern.RatQuad(x_train.shape[1], ARD=True)
    model = GPy.models.GPRegression(x_train, y_train, kernel=k)
    model.optimize(messages=False)
    y_pred, cov = model.predict(x_test)
    return y_pred, cov


def predictor_linreg(data_obj, fold_spec, **kwargs):
    y_true = data_obj.y_test.astype('float')

    model = linear_model.LinearRegression()
    model.fit(data_obj.x_train, data_obj.y_train)
    y_predicted = model.predict(data_obj.x_test)
    cov = 0
    return np.array([y_true[0, 0], y_predicted[0, 0], cov]), model


def predictor_bayesian_linreg(data_obj, fold_spec, **kwargs):
    y_true = data_obj.y_test.astype('float')

    model = linear_model.BayesianRidge()
    model.fit(data_obj.x_train, data_obj.y_train)
    y_predicted = model.predict(data_obj.x_test)
    cov = 0
    return np.array([y_true[0, 0], y_predicted[0], cov]), model


def predictor_greedy(data_obj, fold_spec, **kwargs):

    max_kernel = kwargs.get('max_kernel', 1)
    parallel = kwargs.get('parallel', False)
    num_process = kwargs.get('num_process', None)
    max_mult = kwargs.get('max_mult', 2)

    greedy = search.GreedySearch(data_obj,
                                 None,
                                 max_kernel=max_kernel,
                                 max_mult=max_mult,
                                 parallel=parallel,
                                 num_process=num_process)

    preloaded_model = kwargs.get('preloaded_model', None)

    if preloaded_model is not None:
        with open(preloaded_model, 'rb') as f:
            result = pickle.load(f)
    else:
        model, bic, result = greedy.search()
    result_copy = copy.deepcopy(result)
    best = result_copy.best[0]
    model = best.build_model(data_obj.x_train, data_obj.y_train)
    y_predicted, cov = model.predict(data_obj.x_test)

    y_predicted = y_predicted.astype('float')
    y_test = data_obj.y_test.astype('float')

    return np.hstack([y_test, y_predicted, cov]), result


def predictor_persistence(data_obj, fold_spec, **kwargs):
    y_last = data_obj.y_train[-1, 0]
    y_test = data_obj.y_test
    y_predicted = np.empty(y_test.shape)
    y_predicted.fill(y_last)
    cov = np.empty(y_test.shape)
    cov.fill(0)
    return np.hstack([y_test, y_predicted, cov]), y_predicted


def predictor_nar(data_obj, fold_spec, **kwargs):
    kernel_class = kwargs.get('kernel', GPy.kern.RBF)
    is_restart = kwargs.get('restart', False)
    is_ard = kwargs.get('ard', False)
    kernel = kernel_class(data_obj.x_train.shape[1], ARD=is_ard)
    model = GPy.models.GPRegression(data_obj.x_train, data_obj.y_train,
                                    kernel=kernel)
    if is_restart:
        num_restarts = kwargs.get('num_restarts', 10)
        model.optimize_restarts(num_restarts=num_restarts,
                                robust=True,
                                verbose=False)
    else:
        model.optimize()
    y_predicted, cov = model.predict(data_obj.x_test)
    return (np.array([data_obj.y_test[0, 0], y_predicted[0, 0], cov[0, 0]]),
            model.param_array)


def predictor_narx(data_obj, fold_spec, **kwargs):
    kernel_class = kwargs.get('kernel', GPy.kern.RBF)
    is_restart = kwargs.get('restart', False)
    kernel = kernel_class(data_obj.x_train.shape[1])
    model = GPy.models.GPRegression(data_obj.x_train, data_obj.y_train,
                                    kernel=kernel)
    if is_restart:
        num_restarts = kwargs.get('num_restarts', 10)
        model.optimize_restarts(num_restarts=num_restarts,
                                robust=True,
                                verbose=False)
    else:
        model.optimize()
    y_predicted, cov = model.predict(data_obj.x_test)
    return (np.array([data_obj.y_test[0, 0], y_predicted[0, 0], cov[0, 0]]),
            model.param_array)


def predictor_dummy(data_obj, fold_spec, **kwargs):
    return np.array([[1, 2, 3]])


def predictor_dummy2(data_obj, fold_spec, **kwargs):
    return np.array([[4, 5, 6], [7, 8, 9]])


# from rpy2.robjects import numpy2ri
# from rpy2.robjects.packages import importr
#
# def predictor_arima(data_obj, fold_spec, **kwargs):
#     horizon = fold_spec['horizon']
#
#     numpy2ri.activate()
#     forecast = importr('forecast')
#
#     # get parameter
#     is_stationary = kwargs.get('stationary', False)
#     is_seasonal = kwargs.get('seasonal', True)
#     is_stepwise = kwargs.get('stepwise', False)
#
#     y_train = data_obj.y_train.flatten()
#     fit = forecast.auto_arima(y_train,
#                               stationary=is_stationary,
#                               seasonal=is_seasonal,
#                               stepwise=is_stepwise)
#
#     order = list(fit[6])
#     arima_str = 'ARIMA({0},{5},{1}) ({2},{6},{3})_{4}'
#     model = arima_str.format(*order)
#
#     result = forecast.forecast(fit, h=horizon, level=np.array([0.95]))
#     y_pred = numpy2ri.ri2py(result[3])
#
#     cov = np.empty([horizon, 1])
#     cov.fill(0)
#
#     return np.hstack([data_obj.y_test, y_pred.reshape([-1, 1]), cov]), model
