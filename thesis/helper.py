# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:43:09 2016

@author: bagas
"""

import os
import subprocess
import pandas as pd
import numpy as np
import json
import data
import requests
import socket
from ConfigParser import SafeConfigParser
try:
    import cPickle as pickle
except:
    import pickle

import config


def generate_sbatch_script(metadata):
    s = """#!/bin/sh
#SBATCH --partition=%(job_type)s --qos=%(job_type)s
#SBATCH --time=%(time)s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cpu)s
#SBATCH --mem=%(mem)s
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(output_dir)s
#SBATCH --error=%(error_dir)s
#SBATCH --mail-type=BEGIN,END,FAIL

python %(script)s
    """

    formatted_s = s % metadata
    print formatted_s

    with open('%s.sh' % 'pipe.name', 'w') as f:
        f.write(formatted_s)


def read_config(config_path=None):
    config = None
    if config_path is None:
        if os.path.isfile('./setting.conf'):
            config = SafeConfigParser()
            config.read('./setting.conf')
        else:
            raise RuntimeError('default config file is not found')
    else:
        if os.path.isfile(config):
            config = SafeConfigParser()
            config.read(config)
        else:
            raise RuntimeError('config %s is not found' % config)

    return config


def load_pickle(filename):
    config = read_config()
    path = config.get('folder', 'output') + filename
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def send_email(to, subject, message):
    email_msg = "echo %s" % message
    mail_cmd = 'mail -s %s %s' % (subject, to)
    subprocess.call(email_msg + ' | ' + mail_cmd, shell=True)


def send_mailgun(to, subject, message):
    key = 'key-38910a29f5ca88f0214cd02b803b1bd3'
    domain = 'sandbox669c54c1286147de816804aa007abd9d.mailgun.org'

    request_url = "https://api.mailgun.net/v3/%s/messages" % domain
    request = requests.post(request_url, auth=('api', key), data={
        'from': 'runner@bagas.me',
        'to': to,
        'subject': subject,
        'text': message
    })

    print 'Status: {0}'.format(request.status_code)
    print 'Body:   {0}'.format(request.text)


def get_feature_table(config):
    data_folder = config.get('folder', 'data')
    df = pd.read_csv(data_folder + 'fields.csv', index_col=0)
    return df


def get_feature_array(ids):
    feature_table = get_feature_table(read_config())
    return feature_table.ix[ids, 'name'].values.tolist()


def read_job_runner(json_file, conf):
    with open(json_file, 'r') as f:
        job_spec = json.load(f)

    name = job_spec['name']
    jobs = job_spec['jobs']

    output_folder = conf.get('folder', 'output')

    dfs = []
    names = []
    for job in jobs:
        filename = "%s_%s.csv" % (name, job['name'])
        output_path = output_folder + filename
        df = pd.read_csv(output_path, index_col=[0, 1])
        df.index.names = [None, None]
        dfs.append(df)
        names.append(job['name'])

    df_combined = pd.concat(dfs, keys=names)
    return df_combined


def _generate_start_stop_date(start_date, training_duration,
                              offset=pd.tseries.offsets.Day):

    start_tstamp = pd.Timestamp(start_date)
    finish_tstamp = start_tstamp + offset(training_duration)
    return str(start_tstamp), str(finish_tstamp)


def random_fold(fold_num, weather_data, save=False, fname=None):

    start_stop = []
    for i in range(fold_num):
        random_date = str(pd.Timestamp(np.random.choice(weather_data.index)))
        start_stop_date = _generate_start_stop_date(random_date, 2)
        start_stop.append(start_stop_date)

    if save:
        with open(fname + '.fold', 'wb') as f:
            pickle.dump(start_stop, f)

    return start_stop


def random_fold2(fold_num, save=False, fname=None):

    weather_data = data.Data.prefetch_data2()

    start_stop = []
    for i in range(fold_num):
        random_date = str(pd.Timestamp(np.random.choice(weather_data.index)))
        start_stop.append(random_date)

    if save:
        with open(fname + '.date', 'wb') as f:
            pickle.dump(start_stop, f)

    return start_stop


def fold_season(year, save=False, fname=None, weather_data=None):
    seasons = [(3, 1, 5, 31), (6, 1, 8, 31), (9, 1, 11, 30), (12, 1, 2, 28)]

    if weather_data is None:
        weather_data = data.Data.prefetch_data2()

    folds = []
    for idx, season in enumerate(seasons):
        start_month, start_day, stop_month, stop_day = season
        start = pd.datetime(year, start_month, start_day, 0, 0)
        if idx == 3:
            # it is winter
            stop = pd.datetime(year+1, stop_month, stop_day, 23, 55)
        else:
            stop = pd.datetime(year, stop_month, stop_day, 23, 55)

        sliced_index = weather_data[start:stop].index
        random_date = str(pd.Timestamp(np.random.choice(sliced_index)))
        folds.append(random_date)

    if save:
        with open(fname + '.date', 'wb') as f:
            pickle.dump(folds, f)

    return folds


def generate_training_date(end_date,
                           duration,
                           offset=pd.tseries.offsets.Minute(5)):

    finish_tstamp = pd.Timestamp(end_date)
    start_tstamp = finish_tstamp - duration * offset
    return str(start_tstamp), str(finish_tstamp)


def load_fold(fold_name):
    with open(fold_name, 'rb') as f:
        folds = pickle.load(f)

    return folds


def get_season(season, year, release=1):
    try:
        fold_path = config.data + 'season%d_%d' % (year, release)
    except IOError:
        fold_path = 'season%d_%d' % (year, release)

    f = '%s.date' % fold_path
    if season == 'spring':
        season_id = 0
    elif season == 'summer':
        season_id = 1
    elif season == 'fall' or season == 'autumn':
        season_id = 2
    elif season == 'winter':
        season_id = 3
    else:
        raise NotImplementedError('no %s as season' % season)

    seasons = load_fold(f)
    return seasons[season_id], '%s%d%d' % (season, year, release)


def parse_horizon(arr):
    horizons = []
    for h in arr:
        if isinstance(h, basestring):
            # ranged representation
            start_stop = h.split(':')
            start = int(start_stop[0])
            stop = int(start_stop[1]) + 1
            horizons = horizons + range(start, stop)
        elif isinstance(h, int):
            horizons.append(h)
        else:
            raise RuntimeError('wrong horizon format. It is either "x:y" or x')

    return horizons
