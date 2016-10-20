from thesis import search
from thesis import data
from thesis import config
from thesis import helper
#import GPy
import time
import uuid
import sys

max_kernel = int(1)
parallel = bool(int(0))

# data
#data_obj = data.TSData('airline')
raw_data = data.Data.prefetch_data2(sample=True)
data_obj = data.WindData('2016-01-01 00:00:00',
                           '2016-01-01 23:55:00',
                           '2016-01-02 00:00:00',
                           '2016-01-02 23:55:00',
                           skformat=True,
                           data=raw_data)

tic = time.time()
greedy = search.GreedySearch(data_obj,
                             None,
                             max_kernel=max_kernel,
                             parallel=parallel,
                             num_process=None)

model, bic, result = greedy.search()
toc = time.time() - tic
total_time = '%d minutes %d second' % (toc // 60, toc % 60)

# unique_filename = str(uuid.uuid4())
# with open('%s%s.out' % (config.output, unique_filename), 'w') as f:
#     f.write(str(toc))
#     f.write('\n')
#     f.write('num data: %d' % data_obj.x_train.shape[0])
#     f.write('\n')
#     f.write('max kernel=%d' % max_kernel)
#     f.write('\n')
#     f.write('parallel=%s' % str(parallel))
#     f.write('\n')
#     f.write('time: %s' % total_time)
#     f.write('\n')

result.save('result.pkl')

# load again
result2 = search.SearchResult.from_pickle('result.pkl')
best = result2.best[0]
model2 = best.build_model(data_obj.x_train, data_obj.y_train)

print model
print model2
#y_, cov = model.predict(airline_data.x_test)

    # fig, ax = plt.subplots()
    # ax.plot(airline_data.x_train, airline_data.y_train)
    # ax.plot(airline_data.x_test, airline_data.y_test)
    # ax.plot(airline_data.x_test, y_)
    # fig.show()
