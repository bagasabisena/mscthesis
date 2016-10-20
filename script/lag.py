# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:09:01 2016

@author: bagas
"""
from thesis import data
from thesis import pipeline
from thesis import evaluation as ev
from thesis import helper
import GPy
import numpy as np

conf = helper.read_config()
weather_data = data.WindData.prefetch_data(conf)

lagging_window = 20

def hour_data(p, n=lagging_window):
    wind = data.WindData('2016/1/1',
                          '2016/1/7',
                          '2016/1/8 00:00:00',
                          '2016/1/8 01:00:00',
                          None,
                          None, data=weather_data, skformat=True)
    
    return wind.get_window_repr3(n)

def day_data(p, n=lagging_window):
    wind = data.WindData('2016/1/1',
                          '2016/1/7',
                          '2016/1/8',
                          '2016/1/8',
                          None,
                          None, data=weather_data, skformat=True)
    
    return wind.get_window_repr3(n)

def week_data(p, n=lagging_window):
    wind = data.WindData('2016/1/1',
                         '2016/1/14',
                         '2016/1/15',
                         '2016/1/21',
                         None,
                         None, data=weather_data, skformat=True)
    
    return wind.get_window_repr3(n)
    
def predictor(p):
    X = p.x_train
    y = p.y_train
    X_ = p.x_test
    y_ = p.y_test

    rbf = GPy.kern.RBF(X.shape[1])
    model = GPy.models.GPRegression(X, y, kernel=rbf)
    model.optimize(messages=False, max_f_eval=100)
    y_pred = np.zeros(y_.shape[0]).reshape([-1,1])
    covs = np.zeros(y_.shape[0])
    
    X_ = X_.reshape([-1, X_.shape[0]])
    for i in xrange(y_.shape[0]):
        y_pred[i,0], var = model.predict(X_)
        X_ = np.roll(X_, -1)
        X_[0,-1] = y_pred[i]
        covs[i] = var
    
    return y_pred, covs, model

# L=5
pipe = pipeline.Pipeline2(data=hour_data,
                          predictor=predictor,
                          evaluator=[ev.mean_square_error],
                          output='essential',
                          name='L20_hour')

pipe.start()

pipe = pipeline.Pipeline2(data=day_data,
                          predictor=predictor,
                          evaluator=[ev.mean_square_error],
                          output='essential',
                          name='L20_day')

pipe.start()

pipe = pipeline.Pipeline2(data=week_data,
                          predictor=predictor,
                          evaluator=[ev.mean_square_error],
                          output='essential',
                          name='L20_week')

pipe.start()
#sbatch_metadata = {
#    'job_type':'short',
#    'time': '1:00:00',
#    'cpu': '2',
#    'mem': '4096',
#    'job_name': pipe.name,
#    'output_dir': pipe.config.get('folder', 'output') + '%s.out' % pipe.name,
#    'error_dir': pipe.config.get('folder', 'output') + '%s.err' % pipe.name,
#    'script': os.path.basename(__file__)
#}
#
#def generate_sbatch_script(metadata):
#    s = """#!/bin/sh
##SBATCH --partition=%(job_type)s --qos=%(job_type)s 
##SBATCH --time=%(time)s
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=%(cpu)s
##SBATCH --mem=%(mem)s
##SBATCH --job-name=%(job_name)s
##SBATCH --output=%(output_dir)s
##SBATCH --error=%(error_dir)s
##SBATCH --mail-type=BEGIN,END,FAIL
#
#python %(script)s
#    """
#    
#    formatted_s = s % metadata
#    print formatted_s
#    
#    with open('%s.sh' % pipe.name, 'w') as f:
#        f.write(formatted_s)