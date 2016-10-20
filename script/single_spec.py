import os
import pandas as pd
from thesis import helper
from thesis import data
from thesis import predictor
from collections import OrderedDict
from thesis import evaluation as ev

spec = OrderedDict()
spec['name'] = os.path.basename(__file__).split('.')[0]

# --------- general configuration ----------
spec['horizon'] = 1
spec['fold'], spec['fold_name'] = helper.get_season('winter', 2015)
spec['past_data'] = (1, pd.offsets.Day(1))
spec['shift'] = (0, pd.offsets.Day(1))
spec['save_model'] = True
spec['email'] = 'bagasabisena@gmail.com'
# -------- job configuration -----------
# weather_data = data.Data.prefetch_data2()
spec['job'] = {
    'name': 'narx',
    'data': {
        'fun': data.data_wind,
        'extra': {
            'output_feature': 'WindSpd_{Avg}',
            "input_feature": ['WindSpd_{Avg}', 'WindSpd_{Max}'],
            "l": 5,
            "time_as_feature": False,
            "scale_x": True,
            "scale_y": False,
        }
    },
    'predictor': {
        'fun': predictor.predictor_narx,
        'extra': {
            'restart': True
        }
    }
}
# -------- end configuration ----------
