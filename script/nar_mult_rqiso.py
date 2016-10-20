import os

import GPy
import pandas as pd
from thesis import helper
from thesis import data
from thesis import predictor
from collections import OrderedDict
from thesis import evaluation as ev

spec = OrderedDict()
spec['name'] = os.path.basename(__file__).split('.')[0]

# --------- general configuration ----------
spec['horizon'] = [3, 12, 24]
spec['fold'], spec['fold_name'] = helper.get_season('winter', 2015, release=4)
spec['past_data'] = (7, pd.offsets.Day(1))
spec['shift'] = [(i, pd.offsets.Hour(1)) for i in range(24)]
spec['save_model'] = True
spec['email'] = None
# -------- job configuration -----------
# weather_data = data.Data.prefetch_data2()
spec['job'] = {
    'name': 'rqiso',
    'data': {
        'fun': data.data_horizon_single_1h,
        'extra': {
            'output_feature': 'WindSpd_{Avg}',
            "input_feature": 'WindSpd_{Avg}',
            "l": 7,
            "time_as_feature": False,
            "scale_x": False,
            "scale_y": False,
        }
    },
    'predictor': {
        'fun': predictor.predictor_nar,
        'extra': {
            'restart': True,
            'kernel': GPy.kern.RatQuad,
            'ard': False
        }
    }
}
# -------- end configuration ----------
