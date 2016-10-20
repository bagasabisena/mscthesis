from sklearn.pipeline import Pipeline as SPipeline
import os
from ConfigParser import SafeConfigParser
import time
try:
    import cPickle as pickle
except:
    import pickle
import inspect
import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


class Pipeline(object):
    """
    Define whole pipeline in my thesis work
    """

    def __init__(self, data, transformer, predictor, evaluator):
        self.data = data
        self.transformer = transformer
        self.predictor = predictor
        self.evaluator = evaluator

    def start(self):
        # get data from data pipeline
        x_train = self.data.x_train
        y_train = self.data.y_train
        x_test = self.data.x_test
        y_test = self.data.y_test

        # create sklearn template for processing transformation and predictor
        skpipe = []
        if self.transformer:
            for t in self.transformer:
                skpipe.append((type(t).__name__, t))

#        skpipe = [(type(i).__name__, i) for i in self.transformer]
        skpipe.append((type(self.predictor).__name__, self.predictor))
        pipeline = SPipeline(skpipe)

        # fit and predict
        pipeline.fit(x_train, y_train)
        self.y_pred = pipeline.predict(x_test)

        # evaluate
        self.error = [(i.__name__, i(y_test, self.y_pred[0])) for i in self.evaluator]


class Pipeline2(object):
    """
    Define whole pipeline in my thesis work
    """

    def __init__(self, data, predictor, evaluator, output=None, name=None, config=None):
        self.data = data
        self.predictor = predictor
        self.evaluator = evaluator

        if name is None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.name = timestr
        else:
            self.name = name

        self.config = None
        if config is None:
            if os.path.isfile('./setting.conf'):
                self.config = SafeConfigParser()
                self.config.read('./setting.conf')
            else:
                raise RuntimeError('default config file is not found')
        else:
            if os.path.isfile(config):
                self.config = SafeConfigParser()
                self.config.read(config)
            else:
                raise RuntimeError('config %s is not found' % config)

        self.output = output

        # configure logging
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.logger.info('starting task %s' % self.name)

        self.logger.info('loading data')
        self.x_train, self.y_train, self.x_test, self.y_test = self.data(self)
        self.logger.info('data loaded')

        self.logger.info('start prediction')
        self.y_, self.cov, self.model = self.predictor(self)
        self.logger.info(self.model.__str__())
        self.logger.info('prediction finished')

        self.logger.info('start evaluation')
        self.error = [(i.__name__, i(self.y_test, self.y_)) for i in self.evaluator]
        self.error.append(('ll', self.model.log_likelihood()))

        if self.output is not None:
            if isinstance(self.output, basestring):
                if self.output == 'essential':
                    output = self.output_essential()
                elif self.output == 'all':
                    raise NotImplementedError('all output is not yet implemented')
            elif inspect.isfunction(self.output):
                output = self.output(self)
            else:
                raise RuntimeError('unsupported output format')

            output_dir = self.config.get('folder', 'output')
            output_path = '%s%s.pickle' % (output_dir, self.name)
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)

            self.logger.info('output %s generated' % output_path)

        self.logger.info('task %s finished' % self.name)

    def output_essential(self):
        var_to_write = ('x_test', 'y_test', 'y_', 'error')
        output = {}

        for v in var_to_write:
            output[v] = vars(self)[v]

        output['model'] = self.model.__str__()
        return output


class Pipeline3(object):
    """
    Define whole pipeline in my thesis work
    The data is class of which subclass the Data class
    """

    def __init__(self, data, predictor, evaluator, output=None, name=None, config=None):
        self.data = data
        self.predictor = predictor
        self.evaluator = evaluator

        if name is None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.name = timestr
        else:
            self.name = name

        self.config = None
        if config is None:
            if os.path.isfile('./setting.conf'):
                self.config = SafeConfigParser()
                self.config.read('./setting.conf')
            else:
                raise RuntimeError('default config file is not found')
        else:
            if os.path.isfile(config):
                self.config = SafeConfigParser()
                self.config.read(config)
            else:
                raise RuntimeError('config %s is not found' % config)

        self.output = output

        # configure logging
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.logger.info('starting task %s' % self.name)

        self.logger.info('loading data')
        # get data from data pipeline
        self.x_train = self.data.x_train
        self.y_train = self.data.y_train
        self.x_test = self.data.x_test
        self.y_test = self.data.y_test
        self.logger.info('data loaded')

        self.logger.info('start prediction')
        self.y_, self.cov, _ = self.predictor(self.x_train, self.y_train, self.x_test)
        # self.logger.info(self.model.__str__())
        self.logger.info('prediction finished')

        self.logger.info('start evaluation')
        self.error = [(i.__name__, i(self.y_test, self.y_)) for i in self.evaluator]
        # self.error.append(('ll', self.model.log_likelihood()))

        if self.output is not None:
            if isinstance(self.output, basestring):
                if self.output == 'essential':
                    output = self.output_essential()
                elif self.output == 'all':
                    raise NotImplementedError('all output is not yet implemented')
            elif inspect.isfunction(self.output):
                output = self.output(self)
            else:
                raise RuntimeError('unsupported output format')

            output_dir = self.config.get('folder', 'output')
            output_path = '%s%s.pickle' % (output_dir, self.name)
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)

            self.logger.info('output %s generated' % output_path)

        self.logger.info('task %s finished' % self.name)

    def output_essential(self):
        var_to_write = ('x_test', 'y_test', 'y_', 'error')
        output = {}

        for v in var_to_write:
            output[v] = vars(self)[v]

        output['model'] = self.model.__str__()
        return output


def load_pickle(name):
    with open('%s.pickle' % name, 'rb') as f:
        loaded_data = pickle.load(f)

    return loaded_data
