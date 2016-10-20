from thesis import search
from thesis import data
from thesis import helper
from thesis import evaluation as ev
import time
import GPy

start = '2016-01-01'
finish = '2016-01-03'
output_feature = 'WindSpd_{Avg}'
input_feature = ['WindSpd_{Avg}']

conf = helper.read_config()
weather_data = data.Data.prefetch_data(conf, sample=True)
mvar_data = data.HorizonData(start,
                             finish,
                             h=288,
                             output_feature=output_feature,
                             input_feature=input_feature,
                             l=3,
                             time_as_feature=False,
                             data=weather_data,
                             skformat=True)

tic = time.time()

base = (GPy.kern.RBF, GPy.kern.Linear)
mvar_greedy = search.MultivariateGreedySearch(mvar_data,
                                              None,
                                              max_kernel=2,
                                              parallel=False,
                                              num_process=None)
model, bic, best_list = mvar_greedy.search()
toc = time.time() - tic

print "execution time: %d minute %d second" % (toc // 60, toc % 60)

y_, cov = model.predict(mvar_data.x_test)

print ev.mean_square_error(mvar_data.y_test, y_)

tic = time.time()
model2 = GPy.models.GPRegression(mvar_data.x_train,
                                 mvar_data.y_train,
                                 GPy.kern.RBF(mvar_data.x_train.shape[1]))
model.optimize_restarts(verbose=False)
toc = time.time() - tic
print "execution time: %d minute %d second" % (toc // 60, toc % 60)
y_2, cov = model.predict(mvar_data.x_test)
print ev.mean_square_error(mvar_data.y_test, y_2)

print 'actual: %.4f' % mvar_data.y_test
print 'greedy: %.4f' % y_
print 'rbf: %.4f' % y_2
