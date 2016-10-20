import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
import pandas.tseries.offsets as offset
import config


class Data(object):

    def __init__(self, data=None, config=None):
        # check if data variable is already in the workspace
        if data is not None:
            self.data = data
        else:
            # data not loaded
            self.data = self.get_data()

        self.x_train, self.y_train, self.x_test, self.y_test = self.split()
        self.x_train = self.x_train.astype('float')
        self.y_train = self.y_train.astype('float')
        self.x_test = self.x_test.astype('float')
        self.y_test = self.y_test.astype('float')

    @staticmethod
    def prefetch_data(config=None, station='Capelle', sample=False):
        data_folder = config.get('folder', 'data')
        if sample:
            data = pd.read_csv(data_folder + station + '_sample' + '.csv',
                               parse_dates=['DateTime'],
                               index_col=['DateTime'])
        else:
            data = pd.read_csv(data_folder + station + '.csv',
                               parse_dates=['DateTime'],
                               index_col=['DateTime'])
        return data

    @staticmethod
    def prefetch_data2(station='Capelle', sample=False):
        data_folder = config.data
        if sample:
            data = pd.read_csv(data_folder + station + '_sample' + '.csv',
                               parse_dates=['DateTime'],
                               index_col=['DateTime'])
        else:
            data = pd.read_csv(data_folder + station + '.csv',
                               parse_dates=['DateTime'],
                               index_col=['DateTime'])
        return data

    def get_data(self):
        raise NotImplementedError('not implemented')

    def split(self):
        raise NotImplementedError('not implemented')

    def plot(self, predicted=None):
        fig, ax = plt.subplots()
        ax.plot(self.x_train, self.y_train)
        ax.plot(self.x_test, self.y_test)
        legend_label = ['training', 'test']

        if predicted is not None:
            legend_label.append('predicted')
            ax.plot(self.x_test, predicted)

        ax.legend(legend_label)

        return fig, ax


class WindData(Data):

    def __init__(self,
                 training_start,
                 training_finish,
                 test_start,
                 test_finish,
                 val_start=None,
                 val_stop=None,
                 station='Capelle', skformat=False, data=None, scale_y=False):
        self.station = station
        self.test_finish = test_finish
        self.test_start = test_start
        self.training_finish = training_finish
        self.training_start = training_start
        self.val_start = val_start
        self.val_stop = val_stop
        self.skformat = skformat
        self.scale_y = scale_y

        if data is not None:
            super(WindData, self).__init__(data)
        else:
            super(WindData, self).__init__()

    @staticmethod
    def prefetch_data(config=None, station='Capelle'):
        data_folder = config.get('folder', 'data')
        data = pd.read_csv(data_folder + station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    @staticmethod
    def prefetch_wind(station='Capelle'):
        data_folder = config.data
        data = pd.read_csv(data_folder + station + '_wind' + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])

        s = pd.Series(data['WindSpd_{Avg}'].values, data.index,
                      dtype=np.float32)
        s.index.freq = pd.offsets.Minute(5)
        return s

    @staticmethod
    def prefetch_wind_resampled():
        data_folder = config.data
        resampled = pd.read_csv(data_folder + 'resampled_1h.csv',
                                         parse_dates=['DateTime'],
                                         index_col=['DateTime'])

        resampled.columns = ['WindSpd_{Avg}']
        resampled.index.freq = pd.offsets.Hour(1)
        return resampled

    def get_data(self):
        data_folder = config.data
        data = pd.read_csv(data_folder + self.station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def split(self):
        self.wind_speed = self.data['WindSpd_{Avg}']
        y_train = self.wind_speed[self.training_start:self.training_finish].astype('float')
        if self.scale_y:
            y_train = preprocessing.scale(y_train)

        x_train = np.arange(y_train.size)

        if self.val_start:
            self.y_val = self.wind_speed[self.val_start:self.val_stop].astype('float')

            if self.scale_y:
                self.y_val = preprocessing.scale(self.y_val)

            self.x_val = np.arange(y_train.size, y_train.size + self.y_val.size)
            y_test = self.wind_speed[self.test_start:self.test_finish].astype('float')

            x_test = np.arange(y_train.size + self.y_val.size,
                               y_train.size + self.y_val.size + y_test.size)
        else:
            y_test = self.wind_speed[self.test_start:self.test_finish].astype('float')
            x_test = np.arange(y_train.size,
                               y_train.size + y_test.size)

        if self.scale_y:
                y_test = preprocessing.scale(y_test)

        if self.skformat:
            return x_train.reshape([-1, 1]), y_train.reshape([-1, 1]), x_test.reshape([-1, 1]), y_test.reshape([-1, 1])
        else:
            return x_train, y_train, x_test, y_test

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def get_window_repr(self, window):
        window = window + 1
        X_y = self._rolling_window(self.y_train.ravel(), window)
        X = X_y[:, :-1]
        y = X_y[:, -1].reshape([-1, 1])

        X__y_ = self._rolling_window(self.y_test.ravel(), window)
        X_ = X__y_[:, :-1]
        y_ = X__y_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr2(self, window):
        X = self._rolling_window(self.y_train.ravel(), window)
        y = X[:, -1].reshape([-1, 1])
        X_ = self._rolling_window(self.y_test.ravel(), window)
        y_ = X_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n-window+1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

    def slice_by_horizon(self, horizons):
        idx_ = np.array(horizons) - 1
        self.x_test = self.x_test[idx_, :]
        self.y_test = self.y_test[idx_, :]


class TimeData(Data):

    def __init__(self,
                 variable,
                 training_start,
                 training_finish,
                 test_start,
                 test_finish,
                 val_start=None,
                 val_stop=None,
                 station='Capelle', skformat=False, data=None, scale_y=False):
        self.variable = variable
        self.station = station
        self.test_finish = test_finish
        self.test_start = test_start
        self.training_finish = training_finish
        self.training_start = training_start
        self.val_start = val_start
        self.val_stop = val_stop
        self.skformat = skformat
        self.scale_y = scale_y

        if data is not None:
            super(TimeData, self).__init__(data)
        else:
            super(TimeData, self).__init__()

    def get_data(self):
        data_folder = '/Users/bagas/Dropbox/msc/thesis/project/data/weather/'
        data = pd.read_csv(data_folder + self.station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def split(self):
        feature = self.data[self.variable]
        y_train = feature[self.training_start:self.training_finish].astype('float')
        if self.scale_y:
            y_train = preprocessing.scale(y_train)

        x_train = np.arange(y_train.size)

        if self.val_start:
            self.y_val = feature[self.val_start:self.val_stop].astype('float')

            if self.scale_y:
                self.y_val = preprocessing.scale(self.y_val)

            self.x_val = np.arange(y_train.size, y_train.size + self.y_val.size)
            y_test = feature[self.test_start:self.test_finish].astype('float')

            x_test = np.arange(y_train.size + self.y_val.size,
                               y_train.size + self.y_val.size + y_test.size)
        else:
            y_test = feature[self.test_start:self.test_finish].astype('float')
            x_test = np.arange(y_train.size,
                               y_train.size + y_test.size)

        if self.scale_y:
                y_test = preprocessing.scale(y_test)

        if self.skformat:
            return (x_train.reshape([-1, 1]),
                    y_train.reshape([-1, 1]),
                    x_test.reshape([-1, 1]),
                    y_test.reshape([-1, 1]))
        else:
            return x_train, y_train, x_test, y_test

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def get_window_repr(self, window):
        window = window + 1
        X_y = self._rolling_window(self.y_train.ravel(), window)
        X = X_y[:, :-1]
        y = X_y[:, -1].reshape([-1, 1])

        X__y_ = self._rolling_window(self.y_test.ravel(), window)
        X_ = X__y_[:, :-1]
        y_ = X__y_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr2(self, window):
        X = self._rolling_window(self.y_train.ravel(), window)
        y = X[:, -1].reshape([-1, 1])
        X_ = self._rolling_window(self.y_test.ravel(), window)
        y_ = X_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n-window+1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

    def slice_by_horizon(self, horizons):
        idx_ = np.array(horizons) - 1
        self.x_test = self.x_test[idx_, :]
        self.y_test = self.y_test[idx_, :]



class WindDataRandom(Data):

    def __init__(self,
                 date,
                 sample_n=144,
                 station='Capelle', skformat=False):
        self.station = station
        self.date = date
        self.skformat = skformat

        super(WindDataRandom, self).__init__()

    def get_data(self):
        data_folder = '/Users/bagas/Dropbox/msc/thesis/project/data/weather/'
        data = pd.read_csv(data_folder + self.station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def split(self):
        wind_speed = self.data['WindSpd_{Avg}']
        day = wind_speed[self.date].astype('float')

        y_ = day.values
        x_ = np.arange(y_.size)

        sample_n = 144
        x = np.random.choice(x_, sample_n)
        y = np.array([y_[i] for i in x])

        if self.skformat:
            return x.reshape([-1, 1]), y.reshape([-1, 1]), x_.reshape([-1, 1]), y_.reshape([-1, 1])
        else:
            return x, y, x_, y_


class SinData(Data):

    def __init__(self, n=1000, sample_n=100, skformat=False):
        self.n = n
        self.sample_n = sample_n
        self.skformat = skformat

        super(SinData, self).__init__()

    def get_data(self):
        return 'a'

    def split(self):
        x_ = np.linspace(-10, 10, self.n)
        x = np.random.choice(x_, self.sample_n)
        y = 1 + 0.05 * x + np.sin(x) / x + 0.2 * np.random.randn(self.sample_n)
        y_ = 1 + 0.05 * x_ + np.sin(x_) / x_ + 0.2 * np.random.randn(self.n)
        self.y_true = 1 + 0.05 * x_ + np.sin(x_) / x_

        if self.skformat:
            return x.reshape([-1, 1]), y.reshape([-1, 1]), x_.reshape([-1, 1]), y_.reshape([-1, 1])
        else:
            return x, y, x_, y_


class DummyData(Data):

    def __init__(self, skformat=False):
        self.skformat = skformat
        super(DummyData, self).__init__()

    def get_data(self):
        return 'a'

    def split(self):
        x_ = np.array([6,7,8])
        x = np.array([1,2,3,4,5])
        y = np.array([1,2,3,4,5])
        y_ = np.array([6,7,8])

        if self.skformat:
            return x.reshape([-1, 1]), y.reshape([-1, 1]), x_.reshape([-1, 1]), y_.reshape([-1, 1])
        else:
            return x, y, x_, y_

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n-window+1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

class SinDataForecast(Data):

    def __init__(self, n=1000,
                 training=600,
                 validation=200,
                 testing=200,
                 skformat=False):
        self.n = n
        self.training = training
        self.testing = testing
        self.validation = validation
        self.skformat = skformat

        super(SinDataForecast, self).__init__()

    def get_data(self):
        return 'a'

    def split(self):
        x_total = np.linspace(-10, 10, self.n)
        y_total = 1 + 0.05 * x_total + np.sin(x_total) / x_total + 0.2 * np.random.randn(self.n)

        x = x_total[:self.training]
        y = y_total[:self.training]
        self.x_val = x_total[self.training:self.training + self.validation]
        self.y_val = x_total[self.training:self.training + self.validation]
        x_ = x_total[-self.testing:]
        y_ = y_total[-self.testing:]
        self.y_true = 1 + 0.05 * x_total + np.sin(x_total) / x_total

        if self.skformat:
            return x.reshape([-1, 1]), y.reshape([-1, 1]), x_.reshape([-1, 1]), y_.reshape([-1, 1])
        else:
            return x, y, x_, y_


class SunData(Data):

    def __init__(self,
                 training_start,
                 training_finish,
                 test_start,
                 test_finish,
                 val_start=None,
                 val_stop=None,
                 station='Capelle', skformat=False, data=None, scale_y=False,
                 sunset=False):
        self.station = station
        self.test_finish = test_finish
        self.test_start = test_start
        self.training_finish = training_finish
        self.training_start = training_start
        self.val_start = val_start
        self.val_stop = val_stop
        self.skformat = skformat
        self.scale_y = scale_y
        self.sunset = sunset

        if data is not None:
            super(SunData, self).__init__(data)
        else:
            super(SunData, self).__init__()

    @staticmethod
    def prefetch_data(station='Capelle'):
        data_folder = '/Users/bagas/Dropbox/msc/thesis/project/data/weather/'
        data = pd.read_csv(data_folder + station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def get_data(self):
        data_folder = '/Users/bagas/Dropbox/msc/thesis/project/data/weather/'
        data = pd.read_csv(data_folder + self.station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def split(self):
        self.wind_speed = self.data['SR01Up_{Avg}']
        y_train = self.wind_speed[self.training_start:self.training_finish].astype('float')
        if self.scale_y:
            y_train = preprocessing.scale(y_train)

        x_train = np.arange(y_train.size)

        if self.val_start:
            self.y_val = self.wind_speed[self.val_start:self.val_stop].astype('float')

            if self.scale_y:
                self.y_val = preprocessing.scale(self.y_val)

            self.x_val = np.arange(y_train.size, y_train.size + self.y_val.size)
            y_test = self.wind_speed[self.test_start:self.test_finish].astype('float')

            x_test = np.arange(y_train.size + self.y_val.size,
                               y_train.size + self.y_val.size + y_test.size)
        else:
            y_test = self.wind_speed[self.test_start:self.test_finish].astype('float')
            x_test = np.arange(y_train.size,
                               y_train.size + y_test.size)

        if self.scale_y:
                y_test = preprocessing.scale(y_test)

        if self.sunset:
            train_sunset = self.data.ix[self.training_start:self.training_finish,
                                      'is_dark'].values
            train_combo = [x_train[:, np.newaxis], train_sunset[:, np.newaxis]]
            x_train = np.hstack(train_combo)

            test_sunset = self.data.ix[self.test_start:self.test_finish,
                                      'is_dark'].values
            test_combo = [x_test[:, np.newaxis], test_sunset[:, np.newaxis]]
            x_test = np.hstack(test_combo)

        if self.skformat:
            # if x still a 1D array, make it 2d column vector
            # else, it is already 2d vector (because of sunset)
            if x_train.ndim == 1:
                x_train = x_train.reshape([-1, 1])
                x_test = x_test.reshape([-1, 1])

            return x_train, y_train.reshape([-1, 1]), x_test, y_test.reshape([-1, 1])
        else:
            return x_train, y_train, x_test, y_test

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def get_window_repr(self, window):
        window = window + 1
        X_y = self._rolling_window(self.y_train.ravel(), window)
        X = X_y[:, :-1]
        y = X_y[:, -1].reshape([-1, 1])

        X__y_ = self._rolling_window(self.y_test.ravel(), window)
        X_ = X__y_[:, :-1]
        y_ = X__y_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr2(self, window):
        X = self._rolling_window(self.y_train.ravel(), window)
        y = X[:, -1].reshape([-1, 1])
        X_ = self._rolling_window(self.y_test.ravel(), window)
        y_ = X_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n - window + 1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

    def split_day_night(self):
        """Split dataset by night and day time

        Returns
        -------
        (x_train_day, x_train_night, y_train_day, y_train_night,
                x_test_day, x_test_night, y_test_day, y_test_night)
        """
        if not self.sunset:
            raise RuntimeError('no sunset time feature')

        # split the training data according to daytime or nighttime
        idx_tr_day = np.where(self.x_train[:, 1] == 0)
        idx_tr_night = np.where(self.x_train[:, 1] == 1)
        x_train_day = self.x_train[idx_tr_day]
        x_train_night = self.x_train[idx_tr_night]
        y_train_day = self.y_train[idx_tr_day]
        y_train_night = self.y_train[idx_tr_night]

        # split the test data according to daytime or nighttime
        idx_te_day = np.where(self.x_test[:, 1] == 0)
        idx_te_night = np.where(self.x_test[:, 1] == 1)
        x_test_day = self.x_test[idx_te_day]
        x_test_night = self.x_test[idx_te_night]
        y_test_day = self.y_test[idx_te_day]
        y_test_night = self.y_test[idx_te_night]
        return (x_train_day[:, 0].reshape([-1, 1]),
                x_train_night[:, 0].reshape([-1, 1]),
                y_train_day, y_train_night,
                x_test_day[:, 0].reshape([-1, 1]),
                x_test_night[:, 0].reshape([-1, 1]),
                y_test_day, y_test_night)

    @staticmethod
    def combine_day_night(x_day, x_night, y_day, y_night):
        """Combine day and night data and sort them so we can plot them nicely

        Parameters
        ----------
        x_day : np.array
            2d column vector of numpy array, the time index of day
        x_night : np.array
            2d column vector of numpy array, the time index of night
        y_day : TYPE
            2d column vector of numpy array, the solar value of day
        y_night : TYPE
            2d column vector of numpy array, the solar value of night

        Returns
        -------
        TYPE
            2D array where axis 0 is x and axis 1 is y
        """
        # try plotting
        combo = np.vstack([np.hstack([x_day, y_day]),
                          np.hstack([x_night, y_night])])

        combo_sort = combo[combo[:, 0].argsort()]
        return combo_sort


class HorizonData(Data):
    """Summary
    Class for generating h-step-ahead data.
    Support time, lag, time-lag, multivariate lag, and multivariate time-lag
    model

    Attributes
    ----------
    data_finish : string
        Date string to indicate the last time point to include
    data_start : string
        Date string to indicate the first time point to include
    h : int
        Horizon for h-step-ahead
    input_feature : None, string, or array of string
        Description
    l : TYPE
        Description
    N : TYPE
        Description
    output_feature : TYPE
        Description
    scale_x : TYPE
        Description
    scale_y : TYPE
        Description
    skformat : TYPE
        Description
    station : TYPE
        Description
    time_as_feature : TYPE
        Description
    """
    def __init__(self,
                 data_start,
                 data_finish,
                 h,
                 output_feature,
                 input_feature=None,
                 l=1,
                 time_as_feature=False,
                 station='Capelle',
                 skformat=True,
                 data=None,
                 scale_x=False,
                 scale_y=False,
                 offset=pd.offsets.Minute(5)):

        self.station = station
        self.data_finish = data_finish
        self.data_start = data_start
        self.h = h
        self.skformat = skformat
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.time_as_feature = time_as_feature
        self.l = l
        self.offset = offset

        # check if data variable is already in the workspace

        if data is not None:
            super(HorizonData, self).__init__(data)
        else:
            super(HorizonData, self).__init__()

    @staticmethod
    def prefetch_data(config=None, station='Capelle'):
        data_folder = config.get('folder', 'data')
        data = pd.read_csv(data_folder + station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def get_data(self):
        data_folder = config.data
        data = pd.read_csv(data_folder + self.station + '.csv',
                           parse_dates=['DateTime'],
                           index_col=['DateTime'])
        return data

    def split(self):
        self.data_filter = self.data[self.data_start:self.data_finish]
        self.N = len(self.data_filter.index)

        # get test point in the future
        last_idx = self.data_filter.index[-1]
        horizon_idx = last_idx + self.h * self.offset  # data freq is 1 hour
        y_test = self.data[self.output_feature][horizon_idx]

        if self.skformat:
            y_test = np.array([[y_test]])

        # create appropriate model
        if self.input_feature is None:
            # time model
            x_train = np.arange(self.N)
            y_train = self.data_filter[self.output_feature].astype('float').values
            x_test = x_train[-1] + self.h

            if self.skformat:
                x_train = x_train.reshape([-1, 1])
                y_train = y_train.reshape([-1, 1])
                x_test = np.array([[x_test]])

        elif isinstance(self.input_feature, basestring):
            if self.time_as_feature:
                # time-lag
                x_train, y_train, x_test = self._lag_model()
                N = x_train.shape[0]
                time_index = np.arange(N)
                # should check for skformat
                x_train = np.hstack([x_train, time_index.reshape([-1, 1])])
                time_index_future = time_index[-1] + self.h
                x_test = np.hstack([x_test, np.array([[time_index_future]])])

            else:
                # lag
                feature = self.data_filter[self.output_feature].astype('float')
                # check for nan
                if feature.isnull().any():
                    feature_array = feature.interpolate().values
                else:
                    feature_array = feature.values
                strides = self._rolling_window(feature_array, self.l+self.h)
                x_train = strides[:, 0:self.l]
                y_train = strides[:, -1]
                x_test = strides[-1, -self.l:]

                if self.skformat:
                    y_train = y_train.reshape([-1, 1])
                    x_test = x_test.reshape([1, -1])

        elif isinstance(self.input_feature, list):
            if self.time_as_feature:
                # multivariate time-lag
                pass
            else:
                # multivariate lag
                feature_output = self.data_filter[self.output_feature].astype('float')
                strides_output = self._rolling_window(feature_output, self.l+self.h)
                # x_train = strides[:, 0:self.l]
                y_train = strides_output[:, -1]
                # x_test = strides[-1, -self.l:]
                N_ = y_train.shape[0]

                # iterate input feature
                train_feat = []
                test_feat = []
                for f in self.input_feature:
                    feature = self.data_filter[f].astype('float')
                    # check for nan
                    if feature.isnull().any():
                        feature_array = feature.interpolate().values
                    else:
                        feature_array = feature.values

                    # scale
                    if self.scale_x:
                        feature_array = preprocessing.scale(feature_array)

                    strides = self._rolling_window(feature_array, self.l)
                    x = strides[:N_, :]
                    x_ = strides[-1, :]
                    train_feat.append(x)
                    test_feat.append(x_)

                x_train = np.hstack(train_feat)
                x_test = np.hstack(test_feat)

                if self.skformat:
                    y_train = y_train.reshape([-1, 1])
                    x_test = x_test.reshape([1, -1])
        else:
            raise RuntimeError('wrong combination of model')

        return x_train, y_train, x_test, y_test

        # if self.skformat:
        #     return x_train.reshape([-1, 1]), y_train.reshape([-1, 1]), x_test.reshape([-1, 1]), y_test.reshape([-1, 1])
        # else:
        #     return x_train, y_train, x_test, y_test

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _lag_model(self):
        feature = self.data_filter[self.output_feature].astype('float').values
        strides = self._rolling_window(feature, self.l + self.h)
        x_train = strides[:, 0:self.l]
        y_train = strides[:, -1]
        x_test = strides[-1, -self.l:]

        if self.skformat:
            y_train = y_train.reshape([-1, 1])
            x_test = x_test.reshape([1, -1])

        return x_train, y_train, x_test

    def get_window_repr(self, window):
        window = window + 1
        X_y = self._rolling_window(self.y_train.ravel(), window)
        X = X_y[:, :-1]
        y = X_y[:, -1].reshape([-1, 1])

        X__y_ = self._rolling_window(self.y_test.ravel(), window)
        X_ = X__y_[:, :-1]
        y_ = X__y_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr2(self, window):
        X = self._rolling_window(self.y_train.ravel(), window)
        y = X[:, -1].reshape([-1, 1])
        X_ = self._rolling_window(self.y_test.ravel(), window)
        y_ = X_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n - window + 1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

    def multikernel_setup(self):
        d = len(self.input_feature)
        idx = np.arange(d * self.l)
        idx_split = np.split(idx, d)
        setup = [(self.l, idx_split[i].tolist()) for i in range(d)]
        return setup


class TSData(Data):

    def __init__(self, name, ratio=0.8):
        self.folder = config.ts
        self.ratio = ratio

        assert (name in TSData.get_names()), 'no dataset named "%s"' % name
        self.name = name
        self.f = '%s%s.mat' % (self.folder, self.name)
        super(TSData, self).__init__()

    def split(self):
        X = self.data['X']
        y = self.data['y']

        # split dataset by ratio
        length = X.shape[0]
        split_index = int(np.ceil(length * self.ratio))

        x_train = X[:split_index, :]
        x_test = X[split_index:, :]
        y_train = y[:split_index, :]
        y_test = y[split_index:, :]

        return x_train, y_train, x_test, y_test

    def get_data(self):
        mat_file = loadmat(self.f, appendmat=False)
        return mat_file

    @staticmethod
    def get_names():
        folder = config.ts
        files = [f for f in os.listdir(folder) if f.endswith('.mat')]
        return map(lambda x: x.split('.')[0], files)


class HorizonTSData(Data):
    """Summary
    Class for generating h-step-ahead data.
    Support time, lag, time-lag, multivariate lag, and multivariate time-lag
    model

    Attributes
    ----------
    data_finish : string
        Date string to indicate the last time point to include
    data_start : string
        Date string to indicate the first time point to include
    h : int
        Horizon for h-step-ahead
    input_feature : None, string, or array of string
        Description
    l : TYPE
        Description
    N : TYPE
        Description
    output_feature : TYPE
        Description
    scale_x : TYPE
        Description
    scale_y : TYPE
        Description
    skformat : TYPE
        Description
    station : TYPE
        Description
    time_as_feature : TYPE
        Description
    """
    def __init__(self, name, start, stop, h, l=1, skformat=True):

        self.name = name
        self.h = h
        self.l = l
        self.start = start
        self.stop = stop
        self.skformat = skformat

        self.folder = config.ts

        assert (name in TSData.get_names()), 'no dataset named "%s"' % name
        self.name = name
        self.f = '%s%s.mat' % (self.folder, self.name)

        super(HorizonTSData, self).__init__()

    def get_data(self):
        mat_file = loadmat(self.f, appendmat=False)
        return mat_file

    def split(self):

        X = self.data['X'].flatten()
        y = self.data['y'].flatten()

        data = y
        data_filter = data[self.start:self.stop]

        N = len(data)
        index = np.arange(N)
        index_filter = index[self.start:self.stop]

        # get test point in the future
        last_idx = index_filter[-1]
        horizon_idx = last_idx + self.h
        y_test = data[horizon_idx]

        if self.skformat:
            y_test = np.array([[y_test]])

        # # lag
        # feature = self.data_filter[self.output_feature].astype('float')
        # # check for nan
        # if feature.isnull().any():
        #     feature_array = feature.interpolate().values
        # else:
        #     feature_array = feature.values
        strides = self._rolling_window(data_filter, self.l+self.h)
        x_train = strides[:, 0:self.l]
        y_train = strides[:, -1]
        x_test = strides[-1, -self.l:]

        if self.skformat:
            y_train = y_train.reshape([-1, 1])
            x_test = x_test.reshape([1, -1])

        # # create appropriate model
        # if self.input_feature is None:
        #     # time model
        #     x_train = np.arange(self.N)
        #     y_train = self.data_filter[self.output_feature].astype('float').values
        #     x_test = x_train[-1] + self.h
        #
        #     if self.skformat:
        #         x_train = x_train.reshape([-1, 1])
        #         y_train = y_train.reshape([-1, 1])
        #         x_test = np.array([[x_test]])
        #
        # elif isinstance(self.input_feature, basestring):
        #     if self.time_as_feature:
        #         # time-lag
        #         x_train, y_train, x_test = self._lag_model()
        #         N = x_train.shape[0]
        #         time_index = np.arange(N)
        #         # should check for skformat
        #         x_train = np.hstack([x_train, time_index.reshape([-1, 1])])
        #         time_index_future = time_index[-1] + self.h
        #         x_test = np.hstack([x_test, np.array([[time_index_future]])])
        #
        #     else:
        #         # lag
        #         feature = self.data_filter[self.output_feature].astype('float')
        #         # check for nan
        #         if feature.isnull().any():
        #             feature_array = feature.interpolate().values
        #         else:
        #             feature_array = feature.values
        #         strides = self._rolling_window(feature_array, self.l+self.h)
        #         x_train = strides[:, 0:self.l]
        #         y_train = strides[:, -1]
        #         x_test = strides[-1, -self.l:]
        #
        #         if self.skformat:
        #             y_train = y_train.reshape([-1, 1])
        #             x_test = x_test.reshape([1, -1])
        #
        # elif isinstance(self.input_feature, list):
        #     if self.time_as_feature:
        #         # multivariate time-lag
        #         pass
        #     else:
        #         # multivariate lag
        #         feature_output = self.data_filter[self.output_feature].astype('float')
        #         strides_output = self._rolling_window(feature_output, self.l+self.h)
        #         # x_train = strides[:, 0:self.l]
        #         y_train = strides_output[:, -1]
        #         # x_test = strides[-1, -self.l:]
        #         N_ = y_train.shape[0]
        #
        #         # iterate input feature
        #         train_feat = []
        #         test_feat = []
        #         for f in self.input_feature:
        #             feature = self.data_filter[f].astype('float')
        #             # check for nan
        #             if feature.isnull().any():
        #                 feature_array = feature.interpolate().values
        #             else:
        #                 feature_array = feature.values
        #
        #             # scale
        #             if self.scale_x:
        #                 feature_array = preprocessing.scale(feature_array)
        #
        #             strides = self._rolling_window(feature_array, self.l)
        #             x = strides[:N_, :]
        #             x_ = strides[-1, :]
        #             train_feat.append(x)
        #             test_feat.append(x_)
        #
        #         x_train = np.hstack(train_feat)
        #         x_test = np.hstack(test_feat)
        #
        #         if self.skformat:
        #             y_train = y_train.reshape([-1, 1])
        #             x_test = x_test.reshape([1, -1])
        # else:
        #     raise RuntimeError('wrong combination of model')

        return x_train, y_train, x_test, y_test

        # if self.skformat:
        #     return x_train.reshape([-1, 1]), y_train.reshape([-1, 1]), x_test.reshape([-1, 1]), y_test.reshape([-1, 1])
        # else:
        #     return x_train, y_train, x_test, y_test

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _lag_model(self):
        feature = self.data_filter[self.output_feature].astype('float').values
        strides = self._rolling_window(feature, self.l + self.h)
        x_train = strides[:, 0:self.l]
        y_train = strides[:, -1]
        x_test = strides[-1, -self.l:]

        if self.skformat:
            y_train = y_train.reshape([-1, 1])
            x_test = x_test.reshape([1, -1])

        return x_train, y_train, x_test

    def get_window_repr(self, window):
        window = window + 1
        X_y = self._rolling_window(self.y_train.ravel(), window)
        X = X_y[:, :-1]
        y = X_y[:, -1].reshape([-1, 1])

        X__y_ = self._rolling_window(self.y_test.ravel(), window)
        X_ = X__y_[:, :-1]
        y_ = X__y_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr2(self, window):
        X = self._rolling_window(self.y_train.ravel(), window)
        y = X[:, -1].reshape([-1, 1])
        X_ = self._rolling_window(self.y_test.ravel(), window)
        y_ = X_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n - window + 1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

    def multikernel_setup(self):
        d = len(self.input_feature)
        idx = np.arange(d * self.l)
        idx_split = np.split(idx, d)
        setup = [(self.l, idx_split[i].tolist()) for i in range(d)]
        return setup


class HorizonWindowData(Data):
    """Summary
    Class for generating h-step-ahead data.
    Support time, lag, time-lag, multivariate lag, and multivariate time-lag
    model

    Attributes
    ----------
    data_finish : string
        Date string to indicate the last time point to include
    data_start : string
        Date string to indicate the first time point to include
    h : int
        Horizon for h-step-ahead
    input_feature : None, string, or array of string
        Description
    l : TYPE
        Description
    N : TYPE
        Description
    output_feature : TYPE
        Description
    scale_x : TYPE
        Description
    scale_y : TYPE
        Description
    skformat : TYPE
        Description
    station : TYPE
        Description
    time_as_feature : TYPE
        Description
    """
    def __init__(self, y_train, y_test,
                 h, l=1, skformat=True):

        self.y_test = y_test
        self.y_train = y_train
        self.h = h
        self.l = l
        self.skformat = skformat

        super(HorizonWindowData, self).__init__()

    def get_data(self):
        return None

    def split(self):

        # X = self.data['X'].flatten()
        y = np.concatenate((self.y_train, self.y_test))
        data_filter = self.y_train
        y_test = self.y_test[self.h - 1]

        if self.skformat:
            y_test = np.array([[y_test]])

        strides = self._rolling_window(data_filter, self.l+self.h)
        x_train = strides[:, 0:self.l]
        y_train = strides[:, -1]
        x_test = strides[-1, -self.l:]

        if self.skformat:
            y_train = y_train.reshape([-1, 1])
            x_test = x_test.reshape([1, -1])

        return x_train, y_train, x_test, y_test

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _lag_model(self):
        feature = self.data_filter[self.output_feature].astype('float').values
        strides = self._rolling_window(feature, self.l + self.h)
        x_train = strides[:, 0:self.l]
        y_train = strides[:, -1]
        x_test = strides[-1, -self.l:]

        if self.skformat:
            y_train = y_train.reshape([-1, 1])
            x_test = x_test.reshape([1, -1])

        return x_train, y_train, x_test

    def get_window_repr(self, window):
        window = window + 1
        X_y = self._rolling_window(self.y_train.ravel(), window)
        X = X_y[:, :-1]
        y = X_y[:, -1].reshape([-1, 1])

        X__y_ = self._rolling_window(self.y_test.ravel(), window)
        X_ = X__y_[:, :-1]
        y_ = X__y_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr2(self, window):
        X = self._rolling_window(self.y_train.ravel(), window)
        y = X[:, -1].reshape([-1, 1])
        X_ = self._rolling_window(self.y_test.ravel(), window)
        y_ = X_[:, -1].reshape([-1, 1])
        return X, y, X_, y_

    def get_window_repr3(self, window):
        window = window + 1
        y_comb = np.vstack([self.y_train, self.y_test])
        X_y = self._rolling_window(y_comb.ravel(), window)
        X_total = X_y[:, :-1]
        y_total = X_y[:, -1]
        n = self.x_train.shape[0]
        offset = n - window + 1
        X = X_total[:offset, :]
        y = y_total[:offset].reshape([-1, 1])
        X_ = X_total[offset, :]
        y_ = y_total[offset:].reshape([-1, 1])
        return X, y, X_, y_

    def multikernel_setup(self):
        d = len(self.input_feature)
        idx = np.arange(d * self.l)
        idx_split = np.split(idx, d)
        setup = [(self.l, idx_split[i].tolist()) for i in range(d)]
        return setup


class AnyData(Data):
    def __init__(self, x_train, y_train, x_test, y_test,
                 data=None, config=None):

        self.x_train_ = x_train
        self.y_train_ = y_train
        self.x_test_ = x_test
        self.y_test_ = y_test

        super(AnyData, self).__init__(data, config)

    def split(self):
        return self.x_train_, self.y_train_, self.x_test_, self.y_test_

    def get_data(self):
        return None


def create_window(X, y, max_h, num_window, training_ratio=1.0, fold_gap=1):
    n = X.shape[0]
    max_training = n - max_h - num_window + 1 - (fold_gap - 1)
    total_training = int(np.round(training_ratio * max_training))

    last_points = []
    # very stupid implementation
    last_point = -1 + fold_gap
    for i in np.arange(num_window):
        current_point = last_point - fold_gap
        last_points.append(current_point)
        last_point = current_point

    windows = []
    for point in last_points:
        test_start = point - max_h + 1
        test_stop = point + 1
        test_idx = np.arange(test_start, test_stop)

        train_stop = test_start
        train_start = train_stop - total_training
        train_idx = np.arange(train_start, train_stop)

        x_train = X[train_idx]
        y_train = y[train_idx]
        x_test = X[test_idx]
        y_test = y[test_idx]

        windows.append([x_train, y_train, x_test, y_test])

    return windows


def create_folds(X, y, h, num_folds, fold_gap=1):
    n = X.shape[0]
    max_training = n - h - num_folds + 1 - (fold_gap - 1)
    # total_training = int(np.round(training_ratio * max_training))

    last_points = []
    # very stupid implementation
    last_point = -1 + fold_gap
    for i in np.arange(num_folds):
        current_point = last_point - fold_gap
        last_points.append(current_point)
        last_point = current_point

    windows = []
    for point in last_points:
        test_start = point - h + 1
        test_stop = point + 1
        test_idx = np.arange(test_start, test_stop)

        train_stop = test_start
        # train_start = train_stop - total_training
        # train_idx = np.arange(train_start, train_stop)

        x_train = X[:train_stop]
        y_train = y[:train_stop]
        x_test = X[point]
        y_test = y[point]

        windows.append([x_train, y_train,
                        np.array([x_test]), np.array([y_test])])

    return windows


def create_folds_fixed(X, y, h, num_folds, fold_gap=1):
    n = X.shape[0]
    max_training = n - h - num_folds + 1 - (fold_gap - 1)
    # total_training = int(np.round(training_ratio * max_training))

    last_points = []
    # very stupid implementation
    last_point = -1 + fold_gap
    for i in np.arange(num_folds):
        current_point = last_point - fold_gap
        last_points.append(current_point)
        last_point = current_point

    windows = []
    for point in last_points:
        test_start = point - h + 1
        test_stop = point + 1
        test_idx = np.arange(test_start, test_stop)

        train_stop = test_start
        # train_start = train_stop - total_training
        # train_idx = np.arange(train_start, train_stop)

        x_train = X[:train_stop]
        y_train = y[:train_stop]
        x_test = X[point]
        y_test = y[point]

        windows.append([x_train, y_train,
                        np.array([x_test]), np.array([y_test])])

    # get the last window size, then slice
    window_size = windows[-1][0].shape[0]
    windows_fixed = []

    for window in windows:
        x = window[0][-window_size:]
        y = window[1][-window_size:]
        x_ = window[2]
        y_ = window[3]
        windows_fixed.append([x, y, x_, y_])

    return windows_fixed


def data_wind(start, finish, horizons, runner_spec, **kwargs):
    max_h = max(horizons)
    last_idx = pd.Timestamp(finish)
    test_start = last_idx + pd.tseries.offsets.Minute(5)
    test_finish = last_idx + max_h * pd.tseries.offsets.Minute(5)

    raw_data = kwargs.get('raw_data', None)

    data_obj = WindData(start, finish,
                        str(test_start),
                        str(test_finish),
                        skformat=True,
                        data=raw_data)

    data_obj.slice_by_horizon(horizons)

    return [data_obj]


def data_wind_single(start, finish, horizon, runner_spec, **kwargs):
    last_idx = pd.Timestamp(finish)
    test_start = last_idx + pd.tseries.offsets.Minute(5)
    test_finish = last_idx + horizon * pd.tseries.offsets.Minute(5)

    raw_data = kwargs.get('raw_data', None)

    data_obj = WindData(start, finish,
                        str(test_start),
                        str(test_finish),
                        skformat=True,
                        data=raw_data)

    return data_obj


def data_wind_single_1h(start, finish, horizon, runner_spec, **kwargs):
    last_idx = pd.Timestamp(finish)
    test_start = last_idx + pd.tseries.offsets.Hour(1)
    test_finish = last_idx + horizon * pd.tseries.offsets.Hour(1)

    raw_data = kwargs.get('raw_data', None)
    if raw_data is None:
        raw_data = WindData.prefetch_wind_resampled()

    data_obj = WindData(start, finish,
                        str(test_start),
                        str(test_finish),
                        skformat=True,
                        data=raw_data)

    return data_obj


def data_horizon(start, finish, horizons, runner_spec, **kwargs):

    data_objs = []
    output_feature = kwargs['output_feature']
    input_feature = kwargs.get('input_feature', None)
    l = kwargs['l']
    time_as_feature = kwargs.get('time_as_feature', False)
    raw_data = kwargs.get('raw_data', None)
    scale_x = kwargs.get('scale_x', False)
    scale_y = kwargs.get('scale_y', False)

    for h in horizons:
        data_obj = HorizonData(start, finish, h, output_feature,
                               input_feature=input_feature,
                               l=l,
                               time_as_feature=time_as_feature,
                               skformat=True,
                               data=raw_data,
                               scale_x=scale_x,
                               scale_y=scale_y)
        data_objs.append(data_obj)

    return data_objs


def data_horizon_single(start, finish, horizon, runner_spec, **kwargs):

    output_feature = kwargs['output_feature']
    input_feature = kwargs.get('input_feature', None)
    l = kwargs['l']
    time_as_feature = kwargs.get('time_as_feature', False)
    raw_data = kwargs.get('raw_data', None)
    scale_x = kwargs.get('scale_x', False)
    scale_y = kwargs.get('scale_y', False)

    data_obj = HorizonData(start, finish, horizon, output_feature,
                           input_feature=input_feature,
                           l=l,
                           time_as_feature=time_as_feature,
                           skformat=True,
                           data=raw_data,
                           scale_x=scale_x,
                           scale_y=scale_y)

    return data_obj


def data_horizon_single_1h(start, finish, horizon, runner_spec, **kwargs):

    output_feature = kwargs['output_feature']
    input_feature = kwargs.get('input_feature', None)
    l = kwargs['l']
    time_as_feature = kwargs.get('time_as_feature', False)
    raw_data = kwargs.get('raw_data', None)
    scale_x = kwargs.get('scale_x', False)
    scale_y = kwargs.get('scale_y', False)

    if raw_data is None:
        raw_data = WindData.prefetch_wind_resampled()

    data_obj = HorizonData(start, finish, horizon, output_feature,
                           input_feature=input_feature,
                           l=l,
                           time_as_feature=time_as_feature,
                           skformat=True,
                           data=raw_data,
                           scale_x=scale_x,
                           scale_y=scale_y,
                           offset=pd.offsets.Hour(1))

    return data_obj
