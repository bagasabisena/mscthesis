from thesis.runner import synth

command = dict()
command['horizons'] = [10, 20, 40]
command['num_window'] = 20
command['max_h'] = max(command['horizons'])
names = ['greedy_cv']
command['predictors'] = ['run_greedy_cv']

# 4-kernel
file_names = ['osynthetic_0_1']
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
file_names = ['osynthetic_2_0_1']
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
