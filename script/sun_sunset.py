from thesis import data
from thesis import helper
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_start_stop_date2(start_date, training_duration, testing_duration, offset=pd.tseries.offsets.Day):
    start_tstamp = pd.Timestamp(start_date)
    finish_tstamp = start_tstamp + offset(training_duration)
    start_test = finish_tstamp + pd.tseries.offsets.Minute(5)
    stop_test = start_test + offset(testing_duration)
    return (str(start_tstamp), str(finish_tstamp), str(start_test), str(stop_test))


conf = helper.read_config()
weather_data = data.WindData.prefetch_data(conf)


start_stop_date2 = generate_start_stop_date2('2014-12-31 00:00', 1, 1)

sun_data = data.SunData(start_stop_date2[0], start_stop_date2[1],
                        start_stop_date2[2], start_stop_date2[3],
                        skformat=True, data=weather_data, sunset=True)


solar_ir = weather_data.ix[start_stop_date2[0]:start_stop_date2[1],
                           'SR01Up_{Avg}']


%matplotlib inline
solar_ir.plot()

x = np.linspace(-5, 5)
y = np.sin(x)
plt.plot(x, y)
