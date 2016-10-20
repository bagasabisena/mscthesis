import os
import pandas as pd
from thesis import helper
from thesis import data
from thesis import predictor
from thesis import evaluation as ev

spec = dict()
spec['name'] = os.path.basename(__file__).split('.')[0]

# --------- general configuration ----------
spec['horizons'] = ["1:6", 12, 24, 36]
spec['folds'] = helper.load_fold('random5.date')
spec['past_data'] = (2, pd.offsets.Day(1))
spec['errors'] = [ev.error_squared, ev.error_absolute]

# -------- job configuration -----------
weather_data = data.Data.prefetch_data2()
spec['jobs'] = [
    {
        'name': 'narx',
        'data': {
            'fun': data.data_horizon,
            'extra': {
                'output_feature': 'WindSpd_{Avg}',
                "input_feature": ['WindSpd_{Avg}',
                                  'WindDir_{Avg}',
                                  'Tair_{Avg}',
                                  'RH_{Avg}',
                                  'Rain_{Tot}'],
                "l": 5,
                "time_as_feature": False,
                "scale_x": True,
                "scale_y": False,
                "raw_data": weather_data
            }
        },
        'predictor': {
            'fun': predictor.predictor_narx,
            'extra': {
                'restart': True
            }
        }
    }
]

# -------- end configuration ----------

