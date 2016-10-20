from thesis import data, config
import numpy as np
import GPy

horizons = np.arange(1, 110)
lag = 10

results = []
trues = []
for h in horizons:
    co2_data = data.HorizonTSData('mauna', 0, 436, h, 5)
    k = GPy.kern.RBF(co2_data.x_train.shape[1])
    model = GPy.models.GPRegression(co2_data.x_train, co2_data.y_train, kernel=k)
    model.optimize_restarts(robust=True, verbose=False)
    y_, _ = model.predict(co2_data.x_test)
    results.append(y_[0, 0])
    trues.append(co2_data.y_test[0, 0])

result_arr = np.array(results)
true_arr = np.array(trues)
np.savez(config.output + 'co2_nar', y_pred=result_arr, y_true=true_arr)
np.savez('co2_nar', y_pred=result_arr, y_true=true_arr)

