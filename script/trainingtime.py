import os
import time
from collections import OrderedDict

import GPy
import numpy as np
import pandas as pd

from thesis import config
from thesis import data
from thesis import helper
from thesis import predictor
from thesis.runner import wind

spec = OrderedDict()
spec['name'] = os.path.basename(__file__).split('.')[0]

# --------- general configuration ----------
spec['horizon'] = [12]
spec['fold'], spec['fold_name'] = helper.get_season('winter', 2015, release=4)
spec['past_data'] = (15, pd.offsets.Day(1))
spec['shift'] = [(i, pd.offsets.Hour(1)) for i in range(1)]
spec['save_model'] = False
spec['save_result'] = False
spec['email'] = None
spec['file_log'] = False
spec['stream_log'] = False
# -------- job configuration -----------
spec['job'] = {
    'name': 'greedy',
    'data': {
        'fun': data.data_wind_single_1h
    },
    'predictor': {
        'fun': predictor.predictor_greedy,
        'extra': {
            'max_kernel': 5,
            'max_mult': 2
        }
    }
}
# -------- end configuration ----------

weather_data = data.WindData.prefetch_wind_resampled()


def create_gp_model(kernel=GPy.kern.RBF, ard=False, l=5):
    global weather_data
    predictor_spec = dict()
    predictor_spec['fun'] = predictor.predictor_nar
    predictor_spec['extra'] = dict(restart=True,
                                   kernel=kernel,
                                   ard=ard)

    data_spec = {
        'fun': data.data_horizon_single_1h,
        'extra': {
            'output_feature': 'WindSpd_{Avg}',
            "input_feature": 'WindSpd_{Avg}',
            "l": l,
            "time_as_feature": False,
            "scale_x": False,
            "scale_y": False,
            "raw_data": weather_data
        }
    }

    if kernel == GPy.kern.RBF:
        if ard:
            name = 'rbfard'
        else:
            name = 'rbfiso'
    elif kernel == GPy.kern.RatQuad:
        if ard:
            name = 'rqard'
        else:
            name = 'rqiso'
    else:
        raise NotImplementedError('no kernel {}'.format(str(kernel)))
    job_spec = dict(name=name, data=data_spec, predictor=predictor_spec)
    return job_spec


def create_greedy_model(max_kernel=5, max_mult=2):
    job_spec = {
        'name': 'greedy',
        'data': {
            'fun': data.data_wind_single_1h
        },
        'predictor': {
            'fun': predictor.predictor_greedy,
            'extra': {
                'max_kernel': max_kernel,
                'max_mult': max_mult
            }
        }
    }
    return job_spec


def create_lr_model(l=5):
    global weather_data
    job_spec = {
        'name': 'blr',
        'data': {
            'fun': data.data_horizon_single_1h,
            'extra': {
                'output_feature': 'WindSpd_{Avg}',
                "input_feature": 'WindSpd_{Avg}',
                "l": l,
                "time_as_feature": False,
                "scale_x": False,
                "scale_y": False,
                "raw_data": weather_data
            }
        },
        'predictor': {
            'fun': predictor.predictor_bayesian_linreg
        }
    }

    return job_spec


def create_arima_model():
    job_spec = {
        'name': 'arima',
        'data': {
            'fun': data.data_wind_single_1h
        },
        'predictor': {
            'fun': predictor.predictor_arima,
            'extra': {
                'stationary': False,
                'seasonal': False,
                'stepwise': False
            }
        }
    }

    return job_spec


if __name__ == '__main__':
    pasts = np.arange(100, 1001, 100)
    # pasts = np.arange(1) + 1
    job_specs = [create_arima_model()]

    exec_times = np.empty(shape=(len(pasts), len(job_specs)))

    for i, past in enumerate(pasts):
        spec['past_data'] = (past, pd.offsets.Hour(1))
        for j, js in enumerate(job_specs):
            print('model: {}, data: {}'.format(js['name'], past))
            spec['job'] = js
            t = time.time()
            wind.run(spec)
            elapsed = time.time() - t
            exec_times[i, j] = elapsed

    filename = 'time_{}_{}_arima'.format(pasts[0], pasts[-1])
    np.save(config.output + filename, exec_times)
