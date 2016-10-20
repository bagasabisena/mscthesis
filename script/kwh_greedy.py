from thesis import config
from thesis.runner.synth import run

# example of the dict
# {'fold_gap': 1, 'predictors': ['run_greedy'], 'horizons': [5, 10, 20],
# 'training_ratio': 1.0, 'num_window': 10, 'max_h': 20, 'extras': None,
# 'names': ['dummy'], 'output': 'dummy', 'input': 'synthetic_0_1'}

command = dict()
command['fold_gap'] = 1
command['predictors'] = ['run_greedy']
command['horizons'] = [12, 24, 48]
command['training_ratio'] = 1
command['num_window'] = 24
command['max_h'] = max(command['horizons'])
command['names'] = ['greedy']
command['output'] = 'kwh_greedy'
command['input'] = 'kwh'
command['extras'] = dict(preloaded_model=config.output + 'kwh_greedy.model')
run(command)
