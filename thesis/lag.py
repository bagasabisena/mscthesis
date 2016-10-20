import GPy

import data
import numpy as np


class LagCV(object):
    def __init__(self, lags, eval_fun, data_fun, model_fun, horizon,
                 data_params, model_params, eval_params):
        self.horizon = horizon

        if isinstance(lags, np.ndarray):
            self.lags = lags
        elif isinstance(lags, list):
            self.lags = np.array(lags)
        elif isinstance(lags, int):
            self.lags = np.array([lags])
        else:
            error_msg = '{} lag is not supported'.format(type(lags))
            raise NotImplementedError(error_msg)

        self.data_params = data_params
        self.model_params = model_params
        self.eval_params = eval_params

        # load the function
        self.model_fun = getattr(self, 'model_{}'.format(model_fun))
        self.data_fun = getattr(self, 'data_{}'.format(data_fun))
        self.eval_fun = getattr(self, 'eval_{}'.format(eval_fun))

        self.scores = None

    def eval_bic(self, lag, data_obj):
        model = self.model_fun(lag, data_obj)
        ll = model.log_likelihood()
        bic = -2 * ll + lag * np.log(data_obj.x_train.shape[0])
        return bic

    def eval_cv_mae(self, lag, data_obj):
        X = data_obj.x_train
        y = data_obj.y_train

        num_fold = self.eval_params.get('num_fold', 5)
        folds = data.create_folds(X, y, 1, num_fold)

        abs_error = np.empty(num_fold)
        for i, fold in enumerate(folds):
            fold_data_obj = data.AnyData(*fold)
            model = self.model_fun(lag, fold_data_obj)
            y_pred, _ = model.predict(fold_data_obj.x_test)
            abs_error[i] = np.absolute(y_pred - fold_data_obj.y_test)[0, 0]

        return np.mean(abs_error)

    def data_synth(self, lag):
        window = self.data_params['window']
        _, y_train, _, y_test = window
        data_obj = data.HorizonWindowData(y_train, y_test,
                                          h=self.horizon, l=lag)

        return data_obj

    def data_wind(self, lag):
        start = self.data_params['start']
        stop = self.data_params['stop']
        extras = {
            'output_feature': 'WindSpd_{Avg}',
            "input_feature": 'WindSpd_{Avg}',
            "l": lag,
            "time_as_feature": False,
            "scale_x": False,
            "scale_y": False,
            "raw_data": data.WindData.prefetch_wind_resampled()
        }

        data_obj = data.data_horizon_single_1h(start, stop,
                                               self.horizon, None,
                                               **extras)

        return data_obj

    def model_gp(self, lag, data_obj):
        kernel_cls = self.model_params.get('kernel', GPy.kern.RBF)
        is_ard = self.model_params.get('ard', False)
        is_restart = self.model_params.get('restart', True)

        kernel = kernel_cls(data_obj.x_train.shape[1], ARD=is_ard)
        model = GPy.models.GPRegression(data_obj.x_train, data_obj.y_train,
                                        kernel=kernel)
        if is_restart:
            num_restarts = self.model_params.get('num_restarts', 10)
            model.optimize_restarts(num_restarts=num_restarts,
                                    robust=True,
                                    verbose=False)
        else:
            model.optimize()

        return model

    def run(self, verbose=False):
        def run_individual(lag):
            data_obj = self.data_fun(lag)
            score = self.eval_fun(lag, data_obj)
            if verbose:
                print('lag:{}, score:{}'.format(lag, score))

            return score

        self.scores = map(run_individual, self.lags)



if __name__ == '__main__':
    cv = LagCV(None, 'predictive_ll')
    cv.run()
