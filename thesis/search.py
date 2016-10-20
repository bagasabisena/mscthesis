import GPy

import itertools

import multiprocessing
import numpy as np
import copy
import copy_reg
import types
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle

import data


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class Search(object):

    def __init__(self, data_obj):
        self.data_obj = data_obj
        self.x = data_obj.x_train
        self.y = data_obj.y_train
        self.n = self.x.shape[0]

    def calculate_objective(self, model):
        raise NotImplementedError('not yet implemented')

    def search(self):
        raise NotImplementedError('not yet implemented')


class SearchResult(object):

    def __init__(self):
        self.results = []
        self.best = None
        self.candidates = None
        self.best_iter = 0

    def append(self, elem):
        self.results.append(elem)

    def append_from_candidate(self, candidate):
        def arrange(c):
            structure = c[2]
            model = c[0]
            param_array = model.param_array
            mini_model = MiniModel(structure, param_array=param_array)
            mini_model.param_kernel = model.kern.param_array
            mini_model.param_noise = model.param_array[-1]
            score = c[1]
            return mini_model, score

        arranged = map(arrange, candidate)
        self.append(arranged)

    def _calculate_candidates(self):
        self.candidates = []
        for result_per_iteration in self.results:
            best_per_iter = min(result_per_iteration, key=lambda x: x[1])
            self.candidates.append(best_per_iter)

    def calculate_best(self):
        self._calculate_candidates()
        self.best_iter, self.best = min(enumerate(self.candidates, start=1),
                                        key=lambda x: x[1][1])

    def plot_candidate(self):
        fig, ax = plt.subplots()
        x = np.arange(1, len(self.results)+1)
        y = np.array(map(lambda score: score[1], self.candidates))
        ax.plot(x, y)
        ax.set_xticks(x)
        return fig, ax

    def __str__(self):
        if self.best is None:
            return super(SearchResult, self).__str__()
        else:
            return "%s, score: %.3f, iter:%d" % (self.best[0],
                                                 self.best[1],
                                                 self.best_iter)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class GreedySearch(Search):

    def __init__(self, data_obj, base_kernel, max_kernel=5, max_mult=2,
                 parallel=False, num_process=None,
                 obj_fun='bic'):
        super(GreedySearch, self).__init__(data_obj)
        self.obj_fun = obj_fun
        if base_kernel is None:
            self.base_kernel = (GPy.kern.RBF,
                                GPy.kern.Linear,
                                GPy.kern.StdPeriodic)
        else:
            self.base_kernel = base_kernel

        self.max_mult = max_mult
        self.max_kernel = max_kernel
        self.parallel = parallel
        self.num_process = num_process

    def _count_param(self, model):
        return len(model.param_array)

    def _mult_kernel(self, k):
        # check if kernel is a list
        if isinstance(k, list):
            # then it is a multiplication of kernel inside the kernel
            k_obj = map(lambda x: x(input_dim=1), k)
            return GPy.kern.Prod(k_obj)
        else:
            return k(input_dim=1)

    def _build_kernel(self, kernels):
        new_kernel = map(self._mult_kernel, kernels)
        if len(kernels) == 1:
            return new_kernel[0]
        else:
            additive_kernel = GPy.kern.Add(new_kernel)
            return additive_kernel

    def _count_kernel(self, model):
        # check if single kernel or additive
        count = 0
        kernel = model.kern
        if isinstance(kernel, self.base_kernel):
            count += 1
        elif isinstance(kernel, GPy.kern.Add):
            for k in kernel.parts:
                if isinstance(k, GPy.kern.Prod):
                    count += len(k.parts)
                elif isinstance(k, self.base_kernel):
                    count += 1
                else:
                    pass
        else:
            count += len(kernel.parts)

        return count

    def _count_kernel_from_list(self, kernel_list):
        return len(list(itertools.chain.from_iterable(kernel_list)))

    def _create_kernel_list(self, best_kernels):
        if best_kernels is None:
            # first iteration
            kernel_list = map(lambda x: [x], self.base_kernel)
        else:
            # other iteration
            kernel_list = []
            for k in self.base_kernel:

                # addition model

                kernel_copy = copy.copy(best_kernels)
                kernel_copy.append(k)
                kernel_list.append(kernel_copy)

                # multiplicative model

            for i, component in enumerate(best_kernels):
                for k in self.base_kernel:
                    kernel_copy = copy.deepcopy(best_kernels)
                    # elem = kernel_copy[i]
                    if isinstance(component, list):
                        # then it is a multiplication of kernel inside kernel
                        # append
                        if len(component) < self.max_mult:
                            kernel_copy[i].append(k)
                        else:
                            break
                    else:
                        kernel_copy[i] = [kernel_copy[i], k]

                    kernel_list.append(kernel_copy)

        return kernel_list

    def get_kernel_expr2(self, model):
        kernel = model.kern
        if isinstance(kernel, self.base_kernel):
            expr_str = kernel.name
        elif isinstance(kernel, GPy.kern.Add):
            expr_arr = []
            for k in kernel.parts:
                if isinstance(k, GPy.kern.Prod):
                    kern_str = map(lambda x: x.name, k.parts)
                    expr_arr.append(' X '.join(kern_str))
                elif isinstance(k, self.base_kernel):
                    expr_arr.append(k.name)
                else:
                    pass
            expr_str = ' + '.join(expr_arr)
        else:
            kern_str = map(lambda x: x.name, kernel.parts)
            expr_str = ' X '.join(kern_str)

        return "kernel: %s" % expr_str

    def get_kernel_expr(self, kernel_list):

        def print_kernel(k):
            if isinstance(k, list):
                k_list = map(lambda x: x.__name__, k)
                kern_str = ' X '.join(k_list)
            else:
                kern_str = k.__name__

            return kern_str

        kern_expr = map(print_kernel, kernel_list)
        kern_expr_str = ' + '.join(kern_expr)
        return "kernel: %s" % kern_expr_str

    def _eval_candidate(self, kernel_structure):
        print self.get_kernel_expr(kernel_structure)
        kernel = self._build_kernel(kernel_structure)
        model = GPy.models.GPRegression(self.x, self.y, kernel)
        bic = self.calculate_objective(model)
        return model, bic, kernel_structure

    def calculate_objective(self, model):
        model.optimize_restarts(num_restarts=10,
                                robust=True,
                                verbose=False,
                                parallel=False,
                                num_processes=None)

        ll = model.log_likelihood()
        if self.obj_fun == 'bic':
            num_param = self._count_param(model)
            bic = -2 * ll + num_param * np.log(self.n)
            return bic
        elif self.obj_fun == 'll':
            return -ll
        else:
            error_msg = ('{} objective function '
                         'has not been implemented'.format(self.obj_fun))
            raise NotImplementedError(error_msg)

    def search(self):
        best_list = []
        num_iter = 1
        search_result = SearchResult()

        # first iteration
        print '%d ----------------------------' % num_iter
        kernel_list = self._create_kernel_list(None)
        if self.parallel:
            pool = multiprocessing.Pool(processes=self.num_process)
            mapper = pool.map
        else:
            mapper = map

        candidates = mapper(self._eval_candidate, kernel_list)
        search_result.append_from_candidate(candidates)

        # find the model with the lowest BIC
        best_model, best_score, best_kernel = min(candidates,
                                                  key=lambda x: x[1])
        # add best kernel to the array
        # the kernel added after optimisation, so inherit the optimised param
        # best_kernels.append(best_kernel)
        best_list.append((best_model, best_score, best_kernel))
        num_kernel = 1
        print 'best %s' % self.get_kernel_expr(best_kernel)
        num_iter += 1

        while num_kernel < self.max_kernel:
            print '%d ----------------------------' % num_iter

            kernel_list = self._create_kernel_list(best_kernel)

            if self.parallel:
                pool = multiprocessing.Pool(processes=self.num_process)
                mapper = pool.map
            else:
                mapper = map

            candidates = mapper(self._eval_candidate, kernel_list)
            search_result.append_from_candidate(candidates)

            # find the model with the lowest BIC
            (current_model,
             current_score,
             current_kernel) = min(candidates, key=lambda x: x[1])

            best_model = current_model
            best_score = current_score
            best_kernel = current_kernel
            num_kernel = self._count_kernel(best_model)
            num_iter += 1
            print 'best %s' % self.get_kernel_expr(best_kernel)
            best_list.append((best_model, best_score, best_kernel))

    #        if current_score < best_score:
    #            # the new model is better
    #            best_model = current_model
    #            best_score = current_score
    #            num_kernel = _count_kernel(best_model, base_kernel)
    #            num_iter = num_iter + 1
    #            print 'best %s' % get_kernel_expr(best_model, base_kernel)
    #        else:
    #            # the model prior to this is better, no need to continue
    #            # currently just continue to collect
    #            best_model = current_model
    #            best_score = current_score
    #            num_kernel = _count_kernel(best_model, base_kernel)
    #            num_iter = num_iter + 1
    #            print 'best %s' % get_kernel_expr(best_model, base_kernel)
        search_result.calculate_best()
        return best_model, best_score, search_result


class ExhaustiveSearch(GreedySearch):

    def __init__(self, data_obj, base_kernel, max_kernel=5, max_mult=3,
                 parallel=False, num_process=None, obj_fun='bic'):
        super(ExhaustiveSearch, self).__init__(data_obj, base_kernel,
                                               max_kernel, max_mult, parallel,
                                               num_process, obj_fun)

        self.kernel_combination = self.create_combination()
        self.num_combination = len(self.kernel_combination)

    def create_combination(self):

        all_kernels = []
        # test first iteration
        kernel_list = self._create_kernel_list(None)
        all_kernels.append(kernel_list)

        num_kernel = 1
        while num_kernel < self.max_kernel:
            new_kernel = [self._create_kernel_list(k) for k in kernel_list]
            # flatten multiple list of kernel_list
            new_kernel = list(itertools.chain(*new_kernel))
            # add to list of kernel_list (which still per iteration)
            all_kernels.append(new_kernel)
            # assign the kernel_list with the new kernel for further iteration
            kernel_list = new_kernel
            # count the num kernel (currently just add by 1
            num_kernel += 1

        # for idx, kernel_iter in enumerate(all_kernels):
        #     print idx
        #     for k in kernel_iter:
        #         print self.get_kernel_expr(k)

        all_kernels_flatten = list(itertools.chain(*all_kernels))

        return all_kernels_flatten

    def search(self):
        candidates = map(self._eval_candidate, self.kernel_combination)
        search_result = SearchResult()
        search_result.append_from_candidate(candidates)
        search_result.calculate_best()
        return None, None, search_result


class RandomSearch(ExhaustiveSearch):
    def __init__(self, data_obj, base_kernel, max_kernel=5, max_mult=3,
                 parallel=False, num_process=None):
        super(RandomSearch, self).__init__(data_obj, base_kernel, max_kernel,
                                           max_mult, parallel, num_process)

    def search(self, num_sample=1):
        rand_array = np.random.choice(np.array(self.kernel_combination),
                                      size=num_sample,
                                      replace=False)

        rand_list = rand_array.tolist()
        candidates = map(self._eval_candidate, rand_list)
        search_result = SearchResult()
        search_result.append_from_candidate(candidates)
        search_result.calculate_best()
        return None, None, search_result


class UnivariateKernel(object):
    def __init__(self, cls, dim):
        super(UnivariateKernel, self).__init__()
        self.cls = cls
        self.dim = dim

    def __str__(self):
        return "%s_%d" % (self.cls.__name__, self.dim)

    def instantiate(self):
        return self.cls(input_dim=1, active_dims=[self.dim])


class MiniModel(object):
    def __init__(self, structure, param_array=None):
        super(MiniModel, self).__init__()
        self.structure = structure
        # create using build kernel
        self.kernel = None
        # create using update
        self.param_array = param_array
        self.param_kernel = None
        self.param_noise = None

    def build_kernel(self):

        def multiply_kernel(k):
            # check if kernel is a list
            if isinstance(k, list):
                # then it is a multiplication of kernel inside the kernel
                k_obj = map(lambda x: x(input_dim=1), k)
                return GPy.kern.Prod(k_obj)
            else:
                return k(input_dim=1)

        new_kernel = map(multiply_kernel, self.structure)
        if len(self.structure) == 1:
            self.kernel = new_kernel[0]
            # self.param_array = self.kernel.param_array
        else:
            additive_kernel = GPy.kern.Add(new_kernel)
            self.kernel = additive_kernel
            # self.param_array = self.kernel.param_array

    def build_model(self, X, y):
        self.build_kernel()
        self.optimise()
        model = GPy.models.GPRegression(X, y,
                                        kernel=self.kernel,
                                        noise_var=self.param_noise)
        return model

    def get_kernel_expr(self):
        def print_kernel(k):
            if isinstance(k, list):
                k_list = map(lambda x: x.__name__, k)
                kern_str = ' X '.join(k_list)
            else:
                kern_str = k.__name__

            return kern_str

        kern_expr = map(print_kernel, self.structure)
        kern_expr_str = ' + '.join(kern_expr)
        return "kernel: %s" % kern_expr_str

    def __str__(self):
        return self.get_kernel_expr()

    def update(self, param_array):
        # shorthand to update param_array
        self.param_array = param_array

    def optimise(self):
        """
        'Optimise' using already optimised param array
        :return:
        """
        if self.kernel is None:
            raise RuntimeError('please build the kernel first')
        self.kernel[:] = self.param_kernel


class MultivariateGreedySearch(GreedySearch):

    def __init__(self, data_obj, base_kernel, max_kernel=5, parallel=False,
                 num_process=None):
        super(MultivariateGreedySearch, self).__init__(data_obj, base_kernel,
                                                       max_kernel, parallel,
                                                       num_process)
        self.dim_total = self.x.shape[1]
        self.dim_range = np.arange(self.dim_total)

    def _mult_kernel(self, k):
        # check if kernel is a list
        if isinstance(k, list):
            # then it is a multiplication of kernel inside the kernel
            k_obj = map(lambda x: x.instantiate(), k)
            return GPy.kern.Prod(k_obj)
        else:
            return k.instantiate()

    def get_kernel_expr(self, kernel_list):
        def print_kernel(k):
            if isinstance(k, list):
                k_list = map(lambda x: str(x), k)
                kern_str = ' X '.join(k_list)
            else:
                kern_str = str(k)

            return kern_str

        kern_expr = map(print_kernel, kernel_list)
        kern_expr_str = ' + '.join(kern_expr)
        return "kernel: %s" % kern_expr_str

    def _create_kernel_list(self, best_kernels):

        if best_kernels is None:
            # first iteration
            kernel_list = []
            for base in self.base_kernel:
                for dim in range(self.dim_total):
                    unikernel = UnivariateKernel(base, dim)
                    # each element must be in list from the start
                    # so subsequent action will just need to append to the list
                    kernel_list.append([unikernel])
        else:
            # other iteration
            kernel_list = []
            for k in self.base_kernel:

                # addition model
                for dim in range(self.dim_total):
                    kernel_copy = copy.copy(best_kernels)
                    unikernel = UnivariateKernel(k, dim)
                    kernel_copy.append(unikernel)
                    kernel_list.append(kernel_copy)

                # multiplicative model

            for i, component in enumerate(best_kernels):
                for k in self.base_kernel:
                    for dim in range(self.dim_total):
                        kernel_copy = copy.deepcopy(best_kernels)
                        if isinstance(component, list):
                            # then it is a multiplication of kernel inside kernel
                            # append

                            # kernel_copy[i].append(k)
                            break
                        else:
                            unikernel = UnivariateKernel(k, dim)
                            kernel_copy[i] = [kernel_copy[i], unikernel]

                        kernel_list.append(kernel_copy)

        return kernel_list


class CVSearch(GreedySearch):
    def __init__(self, data_obj, base_kernel, horizon,
                 num_folds, max_kernel=5, max_mult=2,
                 parallel=False, num_process=None, fixed=True,
                 fold_gap=1, eval_fun='mae'):
        super(CVSearch, self).__init__(data_obj, base_kernel, max_kernel,
                                       max_mult, parallel, num_process,
                                       obj_fun='bic')
        self.fixed = fixed
        self.eval_fun = eval_fun
        self.fold_gap = fold_gap
        self.num_folds = num_folds
        self.horizon = horizon

    def _eval_candidate(self, kernel_structure):
        print self.get_kernel_expr(kernel_structure)
        kernel = self._build_kernel(kernel_structure)
        if self.fixed:
            folds = data.create_folds_fixed(self.x.flatten(),
                                            self.y.flatten(),
                                            self.horizon,
                                            self.num_folds,
                                            fold_gap=self.fold_gap)
        else:
            folds = data.create_folds(self.x.flatten(),
                                      self.y.flatten(),
                                      self.horizon,
                                      self.num_folds,
                                      fold_gap=self.fold_gap)

        errors = []
        final_model = None
        for i, fold in enumerate(folds):
            x, y, x_, y_ = fold
            x = x.reshape([-1, 1])
            y = x.reshape([-1, 1])
            x_ = x_.reshape([-1, 1])
            y_ = y_.reshape([-1, 1])
            model = GPy.models.GPRegression(x, y, kernel)
            model.optimize_restarts(num_restarts=5,
                                    robust=True,
                                    verbose=False)

            if i == 0:
                final_model = model

            y_pred, _ = model.predict(x_)
            delta = y_pred - y_

            if self.eval_fun == 'mae':
                error_symmetry_fun = np.absolute
            elif self.eval_fun == 'mse':
                error_symmetry_fun = np.square
            else:
                error_str = 'no error named {}'.format(self.eval_fun)
                raise NotImplementedError(error_str)

            errors.append(error_symmetry_fun(delta))

        return final_model, np.mean(errors), kernel_structure


class CVExhautiveSearch(ExhaustiveSearch):
    def __init__(self, data_obj, base_kernel, horizon,
                 num_folds, max_kernel=5, max_mult=2,
                 parallel=False, num_process=None, fixed=True,
                 fold_gap=1, eval_fun='mae'):
        super(CVExhautiveSearch, self).__init__(data_obj, base_kernel,
                                                max_kernel, max_mult, parallel,
                                                num_process, obj_fun='bic')

        self.fixed = fixed
        self.eval_fun = eval_fun
        self.fold_gap = fold_gap
        self.num_folds = num_folds
        self.horizon = horizon

    def _eval_candidate(self, kernel_structure):
        print self.get_kernel_expr(kernel_structure)
        kernel = self._build_kernel(kernel_structure)
        if self.fixed:
            folds = data.create_folds_fixed(self.x.flatten(),
                                            self.y.flatten(),
                                            self.horizon,
                                            self.num_folds,
                                            fold_gap=self.fold_gap)
        else:
            folds = data.create_folds(self.x.flatten(),
                                      self.y.flatten(),
                                      self.horizon,
                                      self.num_folds,
                                      fold_gap=self.fold_gap)

        errors = []
        final_model = None
        for i, fold in enumerate(folds):
            x, y, x_, y_ = fold
            x = x.reshape([-1, 1])
            y = x.reshape([-1, 1])
            x_ = x_.reshape([-1, 1])
            y_ = y_.reshape([-1, 1])
            model = GPy.models.GPRegression(x, y, kernel)
            model.optimize_restarts(num_restarts=5,
                                    robust=True,
                                    verbose=False)

            if i == 0:
                final_model = model

            y_pred, _ = model.predict(x_)
            delta = y_pred - y_

            if self.eval_fun == 'mae':
                error_symmetry_fun = np.absolute
            elif self.eval_fun == 'mse':
                error_symmetry_fun = np.square
            else:
                error_str = 'no error named {}'.format(self.eval_fun)
                raise NotImplementedError(error_str)

            errors.append(error_symmetry_fun(delta))

        return final_model, np.mean(errors), kernel_structure


# if __name__ == '__main__':
#     k = SingularKernel(GPy.kern.RBF, 0)
#     print k
#     print k.instantiate()

