import sys
import numpy as np
import pandas as pd
import importlib
from thesis import helper
from thesis import config


def to_dataframe(result, fold, horizons, errors):
    column_result = ['y_true', 'y_pred', 'cov']
    column_error = [err_fun.__name__ for err_fun in errors]
    column_combined = column_result + column_error
    iterables = [map(lambda x: x[1], fold), horizons]
    midx = pd.MultiIndex.from_product(iterables)
    result_df = pd.DataFrame(result, index=midx, columns=column_combined)
    return result_df


def job_execute(folds, horizons, job_spec):

    result_overall = []
    for f, fold in enumerate(folds):
        fold_spec = dict()
        fold_spec['name'] = "%s_%d" % (job_spec['name'], f)
        print fold_spec['name']
        fold_spec['job'] = job_spec
        # get data
        start, finish = fold
        data_dict = job_spec['data']
        data_fun = data_dict['fun']
        data_extra = data_dict.get('extra')
        if data_extra is None:
            try:
                data_objs = data_fun(start, finish, horizons, fold_spec)
            except:
                print "data error found in %s" % fold_spec['name']
                continue
        else:
            try:
                data_objs = data_fun(start, finish, horizons, fold_spec,
                                     **data_extra)
            except:
                print "data error found in %s" % fold_spec['name']
                continue

        # get predictor
        result_fold = []
        for data_obj in data_objs:
            predictor_dict = job_spec['predictor']
            predictor_fun = predictor_dict['fun']
            predictor_extra = predictor_dict.get('extra')
            if predictor_extra is None:
                try:
                    result = predictor_fun(data_obj, fold_spec)
                except:
                    print "predictor error found in %s" % fold_spec['name']
                    continue
            else:
                try:
                    result = predictor_fun(data_obj,
                                           fold_spec,
                                           **predictor_extra)
                except:
                    print "predictor error found in %s" % fold_spec['name']
                    continue

            result_fold.append(result)

        result_fold_stacked = np.vstack(result_fold)
        result_overall.append(result_fold_stacked)

    # stacked all result from previous fold
    result_overall_stacked = np.vstack(result_overall)

    return result_overall_stacked


def run():
    # global duration, offset
    runner_module = importlib.import_module(sys.argv[1])
    runner_spec = runner_module.spec
    # parse horizons
    runner_spec['horizons'] = helper.parse_horizon(runner_spec['horizons'])
    # create backward fold
    duration, offset = runner_spec['past_data']
    runner_spec['folds'] = map(
        lambda x: helper.generate_training_date(x, duration, offset),
        runner_spec['folds'])
    # run the job
    for job_spec in runner_spec['jobs']:
        job_spec['name'] = "%s_%s" % (runner_spec['name'], job_spec['name'])
        folds = runner_spec['folds']
        horizons = runner_spec['horizons']
        result = job_execute(folds, horizons, job_spec)
        # slap error to the result
        errors = []
        for err_fun in runner_spec['errors']:
            y_true = result[:, 0]
            y_predicted = result[:, 1]
            err = err_fun(y_true, y_predicted)
            errors.append(err.reshape(-1, 1))

        # stack error horizontally
        errors_stacked = np.hstack(errors)
        # stack error with the result
        result_with_error_stacked = np.hstack([result, errors_stacked])
        result_df = to_dataframe(result_with_error_stacked,
                                 folds,
                                 horizons,
                                 runner_spec['errors'])

        output_path = '%s%s.csv' % (config.output, job_spec['name'])
        result_df.to_csv(output_path)


if __name__ == '__main__':
    run()

