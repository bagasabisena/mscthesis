# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 22:46:22 2016

@author: bagas
"""

from sklearn import metrics
import numpy as np


def mean_square_error(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)


def mean_absolute_error(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)


def median_absolute_error(y_true, y_pred):
    return metrics.median_absolute_error(y_true, y_pred)


def r2_score(y_true, y_pred):
    return metrics.r2_score(y_true, y_pred)


def error_squared(y_true, y_predicted):
    return (y_true - y_predicted) ** 2


def error_absolute(y_true, y_predicted):
    return np.abs(y_true - y_predicted)
