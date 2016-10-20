# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:55:17 2016

@author: bagas
"""

from __future__ import division
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
import GPy


class GaussianProcess(BaseEstimator, RegressorMixin):
    """
    Simple implementation of Gaussian Process
    """

    def __init__(self, mean='zero',
                 cov='sq_exp',
                 param={'lambda': 1, 'sigma': 1},
                 correction=10e-6,
                 val=False,
                 optimize=True):
        self.mean = mean
        self.cov = cov
        self.param = param

        self.means_ = ['zero', 'constant']
        self.covs_ = ['sq_exp', 'linear']

        self.correction = correction

        self.val = val

        self.optimize = optimize

    def _cov_sq_exp(self, a, b, param={'lambda': 1, 'sigma': 1}):
        a_tile = np.tile(a, [b.shape[0], 1]).T
        b_tile = np.tile(b, [a.shape[0], 1])

        cov = a_tile - b_tile
        cov = cov / param['lambda']
        cov = cov*cov
        cov = -0.5 * cov
        cov = np.exp(cov)
        return param['sigma'] * param['sigma'] * cov

    def _mean_zero(self):
        return np.zeros(self.X.shape[0])

    def _compute_loglikelihood(self, param):
        noise = self._estimate_noise()
        K = self.cov_fun_(self.X, self.X, param)
        C = (noise**2) * np.eye(self.X.size) + K
        Cinv = np.linalg.inv(C)
        ll = 0.5 * np.log(np.linalg.det(C)) + 0.5 * np.dot(np.dot(self.y, Cinv), self.y) + (self.y.size / 2) * np.log(2*np.pi)
        return ll

    def _ll(self, var):
        """
        internal loglikelihood calculation for optimization software purpose
        """
        noise = self._estimate_noise()
        param = {'lambda': var[0], 'sigma': var[1]}

        K = self.cov_fun_(self.X, self.X, param)
        C = (noise**2) * np.eye(self.X.size) + K
        L = np.linalg.cholesky(C)
        # Cinv = np.linalg.inv(C)
        # compute L' \ (L \ y)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        det = np.product(np.diag(L)**2)

        a = 0.5 * np.dot(self.y, alpha)
        b = 0.5 * np.log(det)
        ll = a + b + (self.y.size / 2) * np.log(2*np.pi)
        return ll

    def _ll_val(self, var):
        """internal loglikelihood calculation with validation"""
        noise = self._estimate_noise()
        self.param = {'lambda': var[0], 'sigma': var[1]}

        mu, K = self.predict(self.X_val, self.y_val)

        C = (noise**2) * np.eye(self.X_val.size) + K
        L = np.linalg.cholesky(C)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, (self.y_val-mu)))
        det = np.product(np.diag(L)**2)

        a = 0.5 * np.dot(self.y_val, alpha)
        b = 0.5 * np.log(det)
        ll = a + b
        return ll

    def _optimize(self):
        res = None
        if self.val:
            res = minimize(self._ll_val,
                       x0=[self.param['lambda'], self.param['sigma']],
                       method='L-BFGS-B',
                       bounds=((10e-6, None), (10e-6, None)))
        else:
            res = minimize(self._ll,
                           x0=[self.param['lambda'], self.param['sigma']],
                           method='L-BFGS-B',
                           bounds=((10e-6, None), (10e-6, None)))

        print res
        self.optimizer_status = res

        return {'lambda': res.x[0], 'sigma': res.x[1]}

    def _estimate_noise(self):
        return self.correction

    def fit(self, X, y=None):

        if self.val:
            self.X = X[:-300]
            self.X_val = X[-300:]
            self.y = y[:-300]
            self.y_val = y[-300:]
        else:
            self.X = X
            self.y = y

        # get mean function
        # currently only zero mean
        if (self.mean not in self.means_):
            raise Exception('mean function is not supported')

        self.mean_fun_ = self._mean_zero

        # get covariance function
        if (self.cov not in self.covs_):
            raise Exception('covariance function is not supported')

        self.cov_fun_ = self._cov_sq_exp

        self.noise = self._estimate_noise()
        if self.optimize:
            self.param = self._optimize()

        return self

    def predict(self, X, y=None):
        K = self.cov_fun_(self.X, self.X, self.param)
        K_ = self.cov_fun_(self.X, X, self.param).T
        K__ = self.cov_fun_(X, X, self.param)
        C = K + self.noise**2 * np.eye(self.X.size)

        y_star = np.dot(np.dot(K_, np.linalg.inv(C)), self.y)
        cov = K__ - np.dot(np.dot(K_, np.linalg.inv(C)), K_.T)

#        L = np.linalg.cholesky(C)
#        alpha = np.linalg.solve(L.T, np.linalg.solve(L,self.y))
##        v = np.linalg.solve(L, K_)
#        y_star = np.dot(K_, alpha)
##        cov = K__ - np.dot(v,v)
# sublime
        return (y_star, cov)

    def get_optimization_status(self):
        return self.get_optimization_status


class GaussianProcessGPy(BaseEstimator, RegressorMixin):

    def __init__(self, mean='zero',
                 cov='sq_exp',
                 param={'lambda': 1, 'sigma': 1},
                 correction=10e-6,
                 val=False,
                 optimize=True):
        self.mean = mean
        self.cov = cov
        self.param = param

        self.means_ = ['zero', 'constant']
        self.covs_ = ['sq_exp', 'linear']

        self.correction = correction
        self.val = val
        self.optimize = optimize

    def fit(self, X, y):
        self.X = X
        self.y = y
        kernel = GPy.kern.RBF(X.shape[1], variance=self.param['sigma'], lengthscale=self.param['lambda'])
        self.model = GPy.models.GPRegression(X, y, kernel)
        if self.optimize:
            self.model.optimize(messages=True)

    def predict(self, X, y=None):
        return self.model.predict(X, y)
