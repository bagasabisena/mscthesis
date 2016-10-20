import GPy
import numpy as np
import sys
import mkl

x = np.linspace(-100, 100, 1000).reshape([-1, 1])
y = np.sin(x)

kernel = GPy.kern.RBF(1)
model = GPy.models.GPRegression(x, y, kernel)

is_parallel = bool(int(sys.argv[1]))

if is_parallel:
    num_process = mkl.get_max_threads()
    model.optimize_restarts(10, robust=True, verbose=False,
                            parallel=True, num_processes=num_process)
else:
    model.optimize_restarts(10, robust=True, verbose=False,
                            parallel=False, num_processes=None)

