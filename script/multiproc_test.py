import multiprocessing
import numpy as np
import random
import sys


def list_append(id):
    """
    Creates an empty list and then appends a
    random number to the list 'count' number
    of times. A CPU-heavy operation!
    """
    print 'start %d' % id
    count = 10000000
    res = []
    for i in range(count):
        res.append(random.random())


if __name__ == "__main__":
    procs = 4   # Number of processes to create

    parallel = sys.argv[1]

    if parallel == 'true':
        print 'multi core'
        print multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=None)
        mapper = pool.map
    else:
        print 'single core'
        mapper = map

    out = mapper(list_append, np.arange(20))

    print "List processing complete."
