import glob
import tempfile
import uuid
import os
from collections import OrderedDict

import palettable
from scipy import io as sio
import matlab.engine

import itertools
import numpy as np
import pandas as pd
import re

from thesis import config
from ..config import matlab_results, output


class WindResult(object):
    def __init__(self, horizons, error_fun):
        self.error_fun = error_fun
        self.horizons = horizons
        self.result_dict = OrderedDict()
        self.true_df = None
        self.grouped = None
        self.names = set()

    def read_timemodel(self, glob_str, name):
        files = glob.glob(glob_str)
        files = sorted(files, key=lambda x: int(x.split('_')[-2]))
        true_arr = []
        result_arr = []
        cov_arr = []
        for f in files:
            arr = np.load(f)
            true = arr[self.horizons - 1, 0]
            result = arr[self.horizons - 1, 1]
            cov = arr[self.horizons - 1, 2]
            true_arr.append(true)
            result_arr.append(result)
            cov_arr.append(cov)

        true_arr = np.vstack(true_arr)
        result_arr = np.vstack(result_arr)
        cov_arr = np.vstack(cov_arr)

        self.result_dict[name] = dict(true=pd.DataFrame(true_arr,
                                                        columns=self.horizons))
        self.result_dict[name]['result'] = pd.DataFrame(result_arr,
                                                        columns=self.horizons)
        self.result_dict[name]['cov'] = pd.DataFrame(cov_arr,
                                                     columns=self.horizons)

        self.names.add(name)

    def read_lagmodel(self, glob_str, name, num_folds):
        def key_fun(x):
            splitted = x.split('_')
            primary_key = int(splitted[-2])
            secondary_key = int(
                re.search('\d+', splitted[-1].split('.')[0]).group(0))
            return primary_key, secondary_key

        files = glob.glob(glob_str)
        sorted_files = sorted(files, key=key_fun)
        sorted_arr = [np.load(f) for f in sorted_files]
        sorted_arr = np.vstack(sorted_arr)
        sorted_arr_split = np.split(sorted_arr, num_folds)

        true_arr = []
        result_arr = []
        cov_arr = []
        for arr in sorted_arr_split:
            true = arr[:, 0]
            result = arr[:, 1]
            cov = arr[:, 2]
            true_arr.append(true)
            result_arr.append(result)
            cov_arr.append(cov)

        true_arr = np.vstack(true_arr)
        result_arr = np.vstack(result_arr)
        cov_arr = np.vstack(cov_arr)

        self.result_dict[name] = dict(true=pd.DataFrame(true_arr,
                                                        columns=self.horizons))
        self.result_dict[name]['result'] = pd.DataFrame(result_arr,
                                                        columns=self.horizons)
        self.result_dict[name]['cov'] = pd.DataFrame(cov_arr,
                                                     columns=self.horizons)

        self.names.add(name)

    def parse(self):
        # parse the true value and check
        self.check_and_parse_true_value()

        # calculate error
        # also aggregate the error into mean and max
        for name, res_dict in self.result_dict.items():
            df = res_dict['result']
            if self.error_fun == 'MAE':
                error_df = np.absolute(self.true_df - df)
            elif self.error_fun == 'MSE':
                error_df = np.square(self.true_df - df)
            elif self.error_fun == 'MAPE':
                error_df = 100 * np.absolute(np.divide(self.true_df - df,
                                                       self.true_df))
            else:
                error_msg = '{} error undefined'.format(self.error_fun)
                raise NotImplementedError(error_msg)

            res_dict['error'] = error_df

            # aggregate error into mean and max error
            mean_array = error_df.mean()
            max_array = error_df.max()
            agg = [('mean', mean_array), ('max', max_array)]
            agg_df = pd.DataFrame.from_items(agg, orient='index',
                                             columns=self.horizons)
            res_dict['agg'] = agg_df

    def group(self):
        group_dict = dict()
        # for bar chart
        agg_dfs = [v['agg'] for v in self.result_dict.values()]
        bar_dfs = []
        for h in self.horizons:
            bar_array = [df[h].values for df in agg_dfs]
            bar_df = pd.DataFrame(bar_array, index=self.result_dict.keys(),
                                  columns=agg_dfs[0].index)
            bar_dfs.append(bar_df)

        group_dict['bar'] = bar_dfs

        # for box plot
        error_dfs = [v['error'] for v in self.result_dict.values()]
        error_combined_dfs = []
        for h in self.horizons:
            values = [df[h].values for df in error_dfs]
            values_array = np.vstack(values).T
            df = pd.DataFrame(values_array, columns=self.result_dict.keys())
            error_combined_dfs.append(df)

        group_dict['boxplot'] = error_combined_dfs

        # for actual plot vs predicted per horizon
        result_dfs = [val['result'] for val in self.result_dict.values()]
        combined_dfs = []
        for h in self.horizons:
            results_arr = [self.true_df[h].values]
            for df in result_dfs:
                results_arr.append(df[h].values)

            results_arr = np.vstack(results_arr).T
            combined_df = pd.DataFrame(results_arr,
                                       columns=['true'] + self.result_dict.keys(),
                                       index=np.arange(results_arr.shape[0])+1)
            combined_dfs.append(combined_df)

        group_dict['compare'] = combined_dfs

        # table of mean error across all dimension
        mean_series = [df['mean'] for df in bar_dfs]
        mean_df = pd.concat(mean_series, axis=1)
        mean_df.columns = self.horizons
        group_dict['mean'] = mean_df

        # table of improvement
        try:
            persistence_df = mean_df.apply(
                lambda x: 100 * (x['persistence'] - x) / x['persistence'])
            group_dict['persistence'] = persistence_df
        except KeyError:
            print 'no persistence result'

        # for matlab results toolbox
        res_arr = np.dstack(error_combined_dfs)
        res_arr_T = np.transpose(res_arr, axes=(1, 2, 0))
        matlab_results_dict = dict(res=res_arr_T)
        matlab_results_dict['dim1'] = error_combined_dfs[0].columns.values
        matlab_results_dict['dim2'] = self.horizons
        matlab_results_dict['dim3'] = np.arange(res_arr_T.shape[2]) + 1
        matlab_results_dict['name1'] = 'model'
        matlab_results_dict['name2'] = 'horizons'
        matlab_results_dict['name3'] = 'window'
        matlab_results_dict['result_title'] = self.error_fun
        group_dict['matlab_results'] = matlab_results_dict

        self.grouped = group_dict

    def check_and_parse_true_value(self):
        # check if all true values are the same, otherwise we are using different data
        # which makes our analysis invalid
        true_arrays = [value['true'].values for value in
                       self.result_dict.values()]
        equality = [np.allclose(i, j) for i, j in
                    itertools.product(true_arrays, true_arrays)]
        assert (np.array(
            equality).all()), "the true values differs. Did you use different data?"

        self.true_df = self.result_dict.values()[0]['true']

    def matlab_results(self, test_type='ind',
                       fmt='html', display_result=False,
                       strf='%.3f'):
        random_name = str(uuid.uuid4().hex)
        mat_path = '{}/{}.mat'.format(tempfile.gettempdir(), random_name)
        output_path = '{}/{}.out'.format(tempfile.gettempdir(), random_name)

        # save to matfile
        sio.savemat(mat_path,
                    self.grouped['matlab_results'])

        # open matlab engine, run the results
        eng = matlab.engine.start_matlab()
        eng.addpath(matlab_results)
        eng.read_result(mat_path, output_path, fmt, test_type, strf, nargout=0)

        eng.exit()

        with open(output_path, 'r') as f:
            out = f.read()

        # delete temp file
        os.remove(mat_path)
        os.remove(output_path)

        if display_result:
            if fmt == 'text' or fmt == 'latex':
                print out
            elif fmt == 'html':
                from IPython.display import HTML, display
                display(HTML(out))
        else:
            return out

    def __str__(self):
        names_arr = list(self.names)
        if len(names_arr) == 0:
            return super(WindResult, self).__str__()
        else:
            return ', '.join(names_arr)

    def plot_bar(self, fig, axes):
        # bar chart
        for h, ax, df in zip(self.horizons,
                             axes.flatten(),
                             self.grouped['bar']):
            df.plot.bar(ax=ax, color=['#1F77B4', '#FF7F0E'])

        return fig, axes

    def plot_box(self, fig, axes):
        tableau_color = palettable.tableau.Tableau_10.hex_colors
        color = dict(boxes=tableau_color[0],
                     whiskers=tableau_color[0],
                     medians=tableau_color[3],
                     caps=tableau_color[7])

        for h, ax, df in zip(self.horizons,
                             axes.flatten(),
                             self.grouped['boxplot']):
            df.plot.box(ax=ax, color=color)

        return fig, axes


if __name__ == '__main__':
    output_folder = output + 'experiment/wind30/winter4/result/'
    horizons = np.array([3, 12, 24, 48])
    result = WindResult(horizons, 'MAE')
    result.read_lagmodel(output_folder+ 'narwind_blr*.npy', 'BLR', 48)


