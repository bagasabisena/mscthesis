import GPy
import numpy as np
from thesis import config
from thesis import data
from thesis import search

kwh_data = np.load(config.ts + 'kwh.npz')
x = kwh_data['x']
y = kwh_data['y_']
windows = data.create_window(x, y, 48, 24)
horizons = [12, 24, 36, 48]
window = windows[-1]

extras = dict()

x_train, y_train, x_test, y_test = window
data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                        x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
greedy = search.GreedySearch(data_obj,
                             base_kernels,
                             max_kernel=6,
                             max_mult=2,
                             obj_fun='bic')

model, bic, result = greedy.search()
result.save(config.output + 'kwh_greedy.pkl')