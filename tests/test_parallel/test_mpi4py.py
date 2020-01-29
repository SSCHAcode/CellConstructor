from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Settings

from cellconstructor.Settings import ParallelPrint as print

import random

import pytest
import sys, os
import time

def test_parallel():
    """
    Compute pi with parallelization
    """

    random.seed(CC.Settings.get_rank() + time.time())

    def get_pi(x):
        _x_ = random.uniform(-1,1)
        _y_ = random.uniform(-1,1)

        if _x_*_x_ + _y_*_y_ < 1:
            return 1
        return 0

    long_list  = range(1000000)
    t1 = time.time()
    result = CC.Settings.GoParallel(get_pi, long_list, reduce_op = "+")
    t2 = time.time()

    pi = result * 4. / len(long_list)

    print("Rank:", CC.Settings.get_rank())
    print("NPROC:", CC.Settings.GetNProc())
    

    print("Total result: {}".format(pi))
    print("Total time: {} s".format(t2 - t1))



if __name__ == "__main__":
    test_parallel()
