from collections import OrderedDict

import itertools

import palettable
import pandas as pd
import numpy as np
from scipy import io as sio
import uuid
import os
import tempfile
import matlab.engine
from ..config import matlab_results


class SynthResult(object):
    def __init__(self, npz_files, horizons,
                 ordering=None, names=None,
                 parse_result=False,
                 error_fun='absolute'):
        self.names = names
        self.ordering = ordering
        self.horizons = horizons
        self.npz_files = map(lambda x: np.load(x), npz_files)
        self.true_df = self.check_and_parse_true_value()
        self.result_dict = None
        self.grouped = None
        self.parse_result = parse_result
        self.error_fun = error_fun

        if self.parse_result:
            self.parse()
            self.group()

    @staticmethod
    def list_models(npz_files):
        npz_files = map(lambda x: np.load(x), npz_files)
        models = []
        for npz_file in npz_files:
            for f in npz_file.files:
                if f != 'true':
                    models.append(f)
        return models

    def parse(self):

        ordered_result = self.order()
        result_dict = OrderedDict()

        # read the ordered result dict, store as dataframe
        for name, value in ordered_result.items():
            df = pd.DataFrame(value, columns=self.horizons)
            result_dict[name] = {'result': df}

        # calculate error
        # also aggregate the error into mean and max
        for name, res_dict in result_dict.items():
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

        self.result_dict = result_dict

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
            persistence_df = mean_df.apply(lambda x: 100 * (x['persistence']-x) / x['persistence'])
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
        # check if all true values are the same
        # otherwise we are using different data
        # which makes our analysis invalid
        true_arrays = [npz_file['true'] for npz_file in self.npz_files]
        equality = [np.allclose(i, j) for i, j in
                    itertools.product(true_arrays, true_arrays)]
        assert (np.array(
            equality).all()), "the true values differ. Did you use different data?"

        # create dataframe for true data
        true_df = pd.DataFrame(true_arrays[0], columns=self.horizons)
        return true_df

    def order(self):
        results = OrderedDict()
        ordered_result = OrderedDict()
        for npz_file in self.npz_files:
            for k, v in npz_file.items():
                if k != 'true':
                    results[k] = v

        if self.ordering is None:
            self.ordering = results.keys()

        if self.names is None:
            self.names = results.keys()

        for key, name in zip(self.ordering, self.names):
            ordered_result[name] = results[key]
        return ordered_result

    def __str__(self):
        models = []
        for npz_file in self.npz_files:
            for f in npz_file.files:
                if f != 'true':
                    models.append(f)

        return ', '.join(models)

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

    def plot_compare(self, fig, axes):
        pass

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


