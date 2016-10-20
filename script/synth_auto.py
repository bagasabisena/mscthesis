from thesis.runner import synth
import numpy as np

# example of the dict
# {'fold_gap': 1, 'predictors': ['run_greedy'], 'horizons': [5, 10, 20],
# 'training_ratio': 1.0, 'num_window': 10, 'max_h': 20, 'extras': None,
# 'names': ['dummy'], 'output': 'dummy', 'input': 'synthetic_0_1'}

command = dict()
command['horizons'] = [12, 24, 48]
names = ['arima']
command['predictors'] = ['run_auto_arima']
command['num_window'] = 24
command['max_h'] = max(command['horizons'])

command['input'] = 'kwh'
command['names'] = names
command['extras'] = dict(stationary=False, seasonal=True)
command['output'] = 'kwh_arima'
synth.run(command)
