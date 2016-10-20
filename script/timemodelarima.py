import os
import pandas as pd
from thesis import helper
from thesis import data
from thesis import predictor
from collections import OrderedDict

from thesis.runner import wind

spec = OrderedDict()
spec['name'] = os.path.basename(__file__).split('.')[0]

# --------- general configuration ----------
spec['horizon'] = [48]
spec['fold'], spec['fold_name'] = helper.get_season('winter', 2015, release=4)
spec['past_data'] = (60, pd.offsets.Day(1))
spec['shift'] = [(i, pd.offsets.Hour(1)) for i in range(24)]
spec['save_model'] = False
spec['email'] = None
spec['file_log'] = False
spec['stream_log'] = False
# -------- job configuration -----------
spec['job'] = {
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
# -------- end configuration ----------

wind.run(spec)
