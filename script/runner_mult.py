import json
import re
import sys
import traceback
import logging

import numpy as np
import pandas as pd
import importlib

import time

from thesis import helper
from thesis import config

import copy

try:
    import cPickle as pickle
except:
    import pickle


def to_dataframe(result, fold, horizons, errors):
    column_result = ['y_true', 'y_pred', 'cov']
    column_error = [err_fun.__name__ for err_fun in errors]
    column_combined = column_result + column_error
    iterables = [map(lambda x: x[1], fold), horizons]
    midx = pd.MultiIndex.from_product(iterables)
    result_df = pd.DataFrame(result, index=midx, columns=column_combined)
    return result_df


def job_execute(runner_spec, log):
    job_spec = runner_spec['job']
    start, stop = runner_spec['fold']
    horizon = runner_spec['horizon']

    # get data
    data_dict = job_spec['data']
    data_fun = data_dict['fun']
    data_extra = data_dict.get('extra')
    if data_extra is None:
        data_obj = data_fun(start, stop, horizon, runner_spec)
    else:
        data_obj = data_fun(start, stop, horizon, runner_spec, **data_extra)

    # get predictor
    predictor_dict = job_spec['predictor']
    predictor_fun = predictor_dict['fun']
    predictor_extra = predictor_dict.get('extra')
    if predictor_extra is None:
        # inject model_save into data_extra
        predictor_extra = {'save_model': runner_spec['save_model']}
    else:
        # add new key data_save to existing extras
        predictor_extra['save_model'] = runner_spec['save_model']

    try:
        result, model = predictor_fun(data_obj,
                                      runner_spec,
                                      **predictor_extra)

        return result, model
    except:
        exc = traceback.format_exc()
        log.error(exc)
        log.error("predictor error found in %s" % runner_spec['name'])
        exit(1)
    # return result, model


def generate_start_stop_date(end_date, duration_past, offset_past,
                             duration_shift, offset_shift):
    finish_time = pd.Timestamp(end_date)

    # shift first to the future, 0 duration will keep it stays
    shift_time = finish_time + duration_shift * offset_shift

    # then move back for the start of the training data
    start_time = shift_time - duration_past * offset_past

    return str(start_time), str(shift_time)


def extract_shift(shift):
    duration, offset = shift
    offset_str = re.search(r'<([^]]+)>', str(offset)).group(1)
    return '%s' % (duration,)


def create_full_name(runner_spec):
    """

    :param runner_spec:
    :return:
    """

    filename = runner_spec['name']
    job_name = runner_spec['job']['name']
    fold_name = runner_spec['fold_name']
    past_data = extract_shift(runner_spec['past_data'])
    shift = extract_shift(runner_spec['shift'])
    horizon = runner_spec['horizon']

    full_name = '%s_%s_%s_%s_%s_h%d' % (filename, job_name, fold_name,
                                        past_data, shift, horizon)
    return full_name


def get_runner_str(runner_spec):
    spec_copy = runner_spec.copy()
    spec_copy['fold'] = ' : '.join(spec_copy['fold'])
    spec_copy['shift'] = extract_shift(spec_copy['shift'])
    spec_copy['past_data'] = extract_shift(spec_copy['past_data'])
    return json.dumps(spec_copy,
                      indent=2,
                      sort_keys=False,
                      default=lambda obj: str(obj))


def send_email_start(runner_spec, to='bagasabisena@gmail.com'):
    body = """
job started

%s
""" % get_runner_str(runner_spec)

    subject = '[start] %s' % runner_spec['name']

    helper.send_mailgun(to, subject, body)


def send_email_error(runner_spec, traceback, to='bagasabisena@gmail.com'):
    body = """
job error

error message:
%s

""" % (traceback,)

    subject = '[error] %s' % runner_spec['name']

    helper.send_mailgun(to, subject, body)


def send_email_finish(result, runner_spec, to='bagasabisena@gmail.com'):
    body = """
job finished

result: %s
%s
""" % (str(result), get_runner_str(runner_spec))

    subject = '[finish] %s' % runner_spec['name']

    helper.send_mailgun(to, subject, body)


def run(runner_spec):
    # parse horizons
    # create backward fold
    duration_past, offset_past = runner_spec['past_data']
    duration_shift, offset_shift = runner_spec['shift']

    # full_name = create_full_name(runner_spec)
    # runner_spec['name'] = full_name

    runner_spec['fold'] = generate_start_stop_date(runner_spec['fold'],
                                                   duration_past,
                                                   offset_past,
                                                   duration_shift,
                                                   offset_shift)

    # print get_runner_str(runner_spec)
    # command_str = raw_input('confirm running? [y/n]')
    #
    # if command_str != 'y':
    #     print 'exiting'
    #     exit(1)

    # configure logger
    log_path = '%s%s.log' % (config.output, runner_spec['name'])
    log = logging.getLogger('runner')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(console_handler)
    log.setLevel(logging.INFO)

    log.info(get_runner_str(runner_spec))
    # run the job
    tic = time.time()
    if runner_spec['email'] is not None:
        send_email_start(runner_spec)

    try:
        result, model = job_execute(runner_spec, log)
    except Exception:
        exc = traceback.format_exc()
        if runner_spec['email'] is not None:
            send_email_error(runner_spec, exc)
        log.error(exc)
        log.error('job error')
        exit(1)

    toc = time.time() - tic
    total_time = '%d minutes %d second' % (toc // 60, toc % 60)

    if runner_spec['email'] is not None:
        send_email_finish(result, runner_spec)

    result_path = '%s%s' % (config.output, runner_spec['name'])
    np.save(result_path, result)

    if runner_spec['save_model']:
        model_path = '%s%s.model' % (config.output, runner_spec['name'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    log.info('job success')
    log.info('result %s' % str(result))
    log.info('total time: %s (%.3f second)' % (total_time, toc))


def main():
    runner_module = importlib.import_module(sys.argv[1])
    runner_spec = runner_module.spec
    for h in runner_spec['horizon']:
        for s in runner_spec['shift']:
            spec_copy = copy.copy(runner_spec)
            spec_copy['horizon'] = h
            spec_copy['shift'] = s

            full_name = create_full_name(spec_copy)
            spec_copy['name'] = full_name

            run(spec_copy)





if __name__ == '__main__':
    main()
    # run()
