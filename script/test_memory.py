import GPy
import numpy as np
import gc


@profile
def predictor(x_train, y_train, x_test):
    X = x_train
    y = y_train
    X_ = x_test
    # y_ = p.y_test

    rbf = GPy.kern.RBF(X.shape[1])
    model = GPy.models.GPRegression(X, y, kernel=rbf)
    model.optimize(messages=False, max_f_eval=100)
    y_pred, cov = model.predict(X_)
    del model
    return y_pred, cov


@profile
def run():
    x_train = np.linspace(-2, 2).reshape([-1, 1])
    y_train = np.sin(x_train).reshape([-1, 1])
    x_test = np.linspace(2, 3).reshape([-1, 1])

    for i in range(20):
        y_, cov = predictor(x_train, y_train, x_test)
        del y_
        del cov
        gc.collect()


if __name__ == '__main__':
    run()
