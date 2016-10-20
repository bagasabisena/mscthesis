
# coding: utf-8

# In[1]:

from thesis import data, config, lag, helper
from thesis.runner.wind import generate_start_stop_date
import numpy as np
import matplotlib.pyplot as plt
import GPy
import pandas as pd

# In[ ]:


def lag_analysis(filename, lags, model_params, horizon, num_fold=5,
                 num_window=20, verbose=False, data_fun='synth'):
    syn_data = np.load(config.ts + filename)
    x = syn_data['x']
    y = syn_data['y_']

    windows = data.create_window(x, y, horizon, num_window)
    window = windows[-1]

    data_params = dict(window=window)
    eval_params = dict(num_fold=num_fold)
    lag_cv = lag.LagCV(lags, 'cv_mae', data_fun, 'gp', horizon, data_params,
                       model_params, eval_params)
    lag_cv.run(verbose=verbose)
    
    return lag_cv


def lag_analysis_wind(season, year, release, lags, model_params,
                      horizon, num_fold=5, past_day=7, verbose=False):

    fold, _ = helper.get_season(season, year, release=4)
    duration_past, offset_past = (past_day, pd.offsets.Day(1))
    duration_shift, offset_shift = (0, pd.offsets.Hour(1))
    start, stop = generate_start_stop_date(fold,
                                           duration_past,
                                           offset_past,
                                           duration_shift,
                                           offset_shift)
    
    data_params = dict(start=start, stop=stop)
    eval_params = dict(num_fold=num_fold)
    lag_cv = lag.LagCV(lags, 'cv_mae', 'wind', 'gp', horizon,
                       data_params, model_params, eval_params)

    lag_cv.run(verbose=verbose)
    
    return lag_cv


def cv(filename, model_param_array, horizons, lags):
    lag_analyses = np.empty((len(model_param_array), len(horizons), len(lags)))

    for i, model_param in enumerate(model_param_array):
        for j, horizon in enumerate(horizons):
            la = lag_analysis(filename, lags, model_param, horizon, verbose=False)
            lag_analyses[i, j, :] = np.array(la.scores)
    
    return lag_analyses


def cv_wind(season, year, fold, model_param_array, horizons, lags, past_day=7):
    lag_analyses = np.empty((len(model_param_array), len(horizons), len(lags)))

    for i, model_param in enumerate(model_param_array):
        for j, horizon in enumerate(horizons):
            la = lag_analysis_wind(season, year, fold, lags, model_param, horizon, past_day=past_day, verbose=False)
            lag_analyses[i, j, :] = np.array(la.scores)
    
    return lag_analyses


def run():
    model_param_array = [dict(kernel=GPy.kern.RBF, ard=False),
                         dict(kernel=GPy.kern.RBF, ard=True),
                         dict(kernel=GPy.kern.RatQuad, ard=False),
                         dict(kernel=GPy.kern.RatQuad, ard=True)]

    lags = np.arange(20)+1

    # run kwh
    horizons = np.array([12, 24, 48])
    arr = cv('kwh.npz', model_param_array, horizons, lags)
    np.save(config.output + 'lag_kwh', arr)

    # run wind
    arr = cv_wind('winter', 2015, 4, model_param_array,
                  horizons, lags, past_day=7)
    np.save(config.output + 'lag_wind_winter', arr)

if __name__ == '__main__':
    run()
