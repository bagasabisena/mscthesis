from thesis.runner import synth
import numpy as np

# example of the dict
# {'fold_gap': 1, 'predictors': ['run_greedy'], 'horizons': [5, 10, 20],
# 'training_ratio': 1.0, 'num_window': 10, 'max_h': 20, 'extras': None,
# 'names': ['dummy'], 'output': 'dummy', 'input': 'synthetic_0_1'}

command = dict()
command['horizons'] = [5, 10, 20, 40]
command['num_window'] = 20
command['max_h'] = max(command['horizons'])
names = ['blr', 'rbfiso', 'rbfard', 'rqiso', 'rqard']
command['predictors'] = ['run_bayesian_lr',
                         'run_rbfiso',
                         'run_rbfard',
                         'run_rqiso',
                         'run_rqard']

file_names = ['osynthetic_0_1']
lag = 12
for f in file_names:
    command['input'] = f
    command['names'] = names
    command['extras'] = dict(lag=lag)
    command['output'] = 'arf20_{}'.format(f)
    synth.run(command)

file_names = ['osynthetic_2_0_1']
lag = 5
for f in file_names:
    command['input'] = f
    command['names'] = names
    command['extras'] = dict(lag=lag)
    command['output'] = 'arf20_{}'.format(f)
    synth.run(command)
