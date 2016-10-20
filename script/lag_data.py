# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:27:36 2016

@author: bagas
"""

X, y, X_, y_ = wind_data_day.get_window_repr(5)

rbf = GPy.kern.RBF(X.shape[1])
model = GPy.models.GPRegression(X, y, kernel=rbf)
model.optimize(messages=True, max_f_eval=100)
y_pred, _ = model.predict(X_)

plt.figure(1, figsize=(10,8))
plt.plot(np.arange(y_pred.shape[0]), y_pred.ravel())
plt.plot(np.arange(y_pred.shape[0]), y_.ravel())
plt.legend(['Predicted GP regression', 'Actual'])
print 'loglikelihood: %f' % model.log_likelihood()
print 'MSE: %f' % ev.mean_square_error(y_, y_pred)

X, y, X_, y_ = wind_data_week.get_window_repr(5)

rbf = GPy.kern.RBF(X.shape[1])
model = GPy.models.GPRegression(X, y, kernel=rbf)
model.optimize(messages=True, max_f_eval=100)

y_pred, _ = model.predict(X_)

plt.figure(1, figsize=(20,8))
plt.xlim(xmax=2100)
plt.plot(np.arange(y_pred.shape[0]), y_pred.ravel())
plt.plot(np.arange(y_pred.shape[0]), y_.ravel())
plt.legend(['Predicted GP regression', 'Actual'])
print 'loglikelihood: %f' % model.log_likelihood()
print 'MSE: %f' % ev.mean_square_error(y_, y_pred)