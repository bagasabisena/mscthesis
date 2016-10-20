import os

import GPy
import pandas as pd
from thesis import helper
from thesis import data
from thesis import predictor
from collections import OrderedDict
import numpy as np
from thesis.runner import wind
from thesis import evaluation as ev

spec = OrderedDict()
spec['name'] = os.path.basename(__file__).split('.')[0]

# --------- general configuration ----------
spec['horizon'] = [48]
spec['fold'], spec['fold_name'] = helper.get_season('winter', 2015, release=4)
spec['past_data'] = (45, pd.offsets.Day(1))
spec['shift'] = [(i, pd.offsets.Hour(1)) for i in range(1)]
spec['save_model'] = False
spec['file_log'] = False
spec['stream_log'] = True
spec['email'] = None
# -------- job configuration -----------
weather_data = data.WindData.prefetch_wind_resampled()
spec['job'] = {
    'name': 'gpiso',
    'data': {
        'fun': data.data_horizon_single_1h,
        'extra': {
            'output_feature': 'WindSpd_{Avg}',
            "input_feature": 'WindSpd_{Avg}',
            "l": 5,
            "time_as_feature": False,
            "scale_x": False,
            "scale_y": False,
            "raw_data": weather_data
        }
    },
    'predictor': {
        'fun': predictor.predictor_nar,
        'extra': {
            'restart': True
        }
    }
}
# -------- end configuration ----------


def create_gp_model(kernel=GPy.kern.RBF, ard=False):
    predictor_spec = dict()
    predictor_spec['fun'] = predictor.predictor_nar
    predictor_spec['extra'] = dict(restart=True,
                                   kernel=kernel,
                                   ard=ard)
    return predictor_spec


models = OrderedDict()
models['rbfiso'] = create_gp_model(GPy.kern.RBF, ard=False)
models['rbfard'] = create_gp_model(GPy.kern.RBF, ard=True)
# models['rqiso'] = create_gp_model(GPy.kern.RatQuad, ard=False)
# models['rqard'] = create_gp_model(GPy.kern.RatQuad, ard=True)
models['blr'] = dict(fun=predictor.predictor_bayesian_linreg)

if __name__ == '__main__':
    # lags = np.array([[10, 8, 13],
    #                  [8, 8, 6],
    #                  [11, 14, 4],
    #                  [9, 13, 4],
    #                  [8, 8, 6]])
    lags = np.array([[10, 8, 13],
                     [8, 8, 6],
                     [8, 8, 6]])
    horizons = [24]
    for i, (model_name, predictor) in enumerate(models.items()):
        for j, horizon in enumerate(horizons):
            print('{}_{}'.format(model_name, horizon))
            spec['horizon'] = [horizon]
            job_spec = spec['job']
            job_spec['name'] = '{}'.format(model_name)
            job_spec['predictor'] = predictor
            spec_data = job_spec['data']
            spec_data['extra']['l'] = lags[i, j]
            wind.run(spec)
