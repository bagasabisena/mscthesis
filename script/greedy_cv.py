from thesis.runner import synth

command = dict()
command['horizons'] = [10, 20, 40]
command['num_window'] = 1
command['max_h'] = max(command['horizons'])
names = ['greedy_cv', 'exhaustive_cv']
command['predictors'] = ['run_greedy_cv', 'run_exhaustive_cv']

# 4-kernel
file_names = ['synthetic_1', 'synthetic_10']
for f in file_names:
    command['input'] = f
    command['names'] = names
    command['extras'] = dict(max_kernel=4,
                             max_mult=2,
                             num_folds=5,
                             fixed=True,
                             eval_fun='mae')
    command['output'] = 'gcv_{}'.format(f)
    synth.run(command)

# 2-kernel
file_names = ['synthetic_2_0_1', 'synthetic_2_1']
for f in file_names:
    command['input'] = f
    command['names'] = names
    command['extras'] = dict(max_kernel=2,
                             max_mult=2,
                             num_folds=5,
                             fixed=True,
                             eval_fun='mae')
    command['output'] = 'gcv_{}'.format(f)
    synth.run(command)
