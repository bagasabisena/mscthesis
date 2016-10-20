from thesis.runner.synth import run
import numpy as np

# models = ['run_rbfiso', 'run_rbfard', 'run_rqiso',
#           'run_rqard', 'run_bayesian_lr']
# names = ['rbfiso', 'rbfard', 'rqiso', 'rqard', 'blr']
models = ['run_bayesian_lr']
names = ['blr']
horizons = [12, 24, 48]

lags = np.array([[7, 7, 9]])

num_window = 24
max_h = max(horizons)
input_file = 'kwh'

for i, (model, name) in enumerate(zip(models, names)):
    for j, horizon in enumerate(horizons):
        command = dict()
        command['horizons'] = [horizon]
        command['num_window'] = num_window
        command['max_h'] = max_h
        command['names'] = [name]
        command['predictors'] = [model]
        command['input'] = input_file
        lag = lags[i, j]
        command['extras'] = dict(lag=lag)
        output_str = 'kwh_{}_{}'.format(name, horizon)
        command['output'] = output_str
        print(output_str)
        run(command)
