import GPy

from thesis import data, search


def test_exhaustive_search():

    data_obj = data.DummyData(skformat=True)
    base_kernel = (GPy.kern.RBF, GPy.kern.Linear)

    exhaustive = search.ExhaustiveSearch(data_obj,
                                         base_kernel=base_kernel,
                                         max_kernel=2,
                                         max_mult=10)

    combination = exhaustive.create_combination()
    assert len(combination) == 10

    _, _, result = exhaustive.search()
    assert isinstance(result, search.SearchResult)
    assert len(result.results[0]) == 10
    assert isinstance(result.best[0], search.MiniModel)


def test_random_search():
    data_obj = data.DummyData(skformat=True)
    base_kernel = (GPy.kern.RBF, GPy.kern.Linear)

    random = search.RandomSearch(data_obj,
                                 base_kernel=base_kernel,
                                 max_kernel=2,
                                 max_mult=10)

    _, _, result = random.search(num_sample=5)
    assert len(result.results[0]) == 5
