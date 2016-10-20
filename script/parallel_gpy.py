import GPy
import sys
import numpy as np

from thesis import data
import multiprocessing


def train(arr):
    x = np.linspace(-2, 2).reshape([-1, 1])
    y = np.sin(x)

    # x = np.load(arr[1])
    # y = np.load(arr[2])

    print 'train kernel %s' % arr[0].name
    model = GPy.models.GPRegression(x, y, arr[0])
    # model.optimize(ipython_notebook=False)
    model.optimize_restarts(5, robust=True, verbose=False)
    return model.log_likelihood()


if __name__ == '__main__':
    # airline = data.AirlineData()
    #
    # x = airline.x_train
    # y = airline.y_train

    xfile = ['airx%d.npy' % i for i in range(4)]
    yfile = ['airy%d.npy' % i for i in range(4)]

    kernels = [GPy.kern.RatQuad(1),
               GPy.kern.StdPeriodic(1),
               GPy.kern.RBF(1),
               GPy.kern.Linear(1)]

    parallel = sys.argv[1]

    if parallel == 'true':
        print 'multi core'
        num_cpu = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_cpu)
        out = pool.map(train, zip(kernels, xfile, yfile))
    else:
        print 'single core'
        out = map(train, zip(kernels, xfile, yfile))

    print out





