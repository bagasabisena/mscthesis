import GPy
import numpy as np

from thesis import data
from thesis import config
from thesis import search


def run_greedy(window, extra_param):
    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    greedy = search.GreedySearch(data_obj,
                                 base_kernels,
                                 max_kernel=extra_param.get('max_kernel', 4),
                                 max_mult=extra_param.get('max_mult', 2),
                                 obj_fun=extra_param.get('obj_fun', 'bic'))

    model, bic, result = greedy.search()
    result.save(config.output + extra_param['name'] + '_greedy_bic4.pkl')


def run_exhaustive(window, extra_param):
    x_train, y_train, x_test, y_test = window
    data_obj = data.AnyData(x_train.reshape([-1, 1]), y_train.reshape([-1, 1]),
                            x_test.reshape([-1, 1]), y_test.reshape([-1, 1]))
    base_kernels = (GPy.kern.RBF, GPy.kern.Linear, GPy.kern.StdPeriodic)
    exhaustive = search.ExhaustiveSearch(data_obj,
                                         base_kernels,
                                         max_kernel=extra_param.get(
                                             'max_kernel', 4),
                                         max_mult=extra_param.get('max_mult',
                                                                  2),
                                         obj_fun=extra_param.get('obj_fun',
                                                                 'bic'))

    model, bic, result = exhaustive.search()
    result.save(config.output + extra_param['name'] + '_ex_ll.pkl')


def runner(filename, max_kernel):
    extra_param = dict(max_kernel=max_kernel, max_mult=2, obj_fun='bic',
                       name=filename.split('.')[0])
    synth_data = np.load(config.ts + filename)
    x = synth_data['x']
    y = synth_data['y_']
    windows = data.create_window(x, y, 20, 10)
    # greedy
    run_greedy(windows[-1], extra_param)
    # exhaustive
    #run_exhaustive(windows[-1], extra_param)


def main():
    filenames = ['synthetic_2_0_1.npz']

    max_kernels = [4]
    for f, max_k in zip(filenames, max_kernels):
        runner(f, max_k)

if __name__ == '__main__':
    main()
