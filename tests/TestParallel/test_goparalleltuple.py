import sys
import os

import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.Settings


# Detect mpi4py availability (used by skipif decorators below).
_has_mpi4py = False
try:
    import mpi4py  # noqa: F401
    _has_mpi4py = True
except ImportError:
    pass


def test_goparalleltuple_serial_returns_all_elements():
    def func(x):
        return [np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])]

    result = CC.Settings.GoParallelTuple(func, [[1, 2]], "+")
    assert len(result) == 2, \
        "GoParallelTuple serial returned %d elements, expected 2" % len(result)
    np.testing.assert_array_equal(result[0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result[1], np.array([[3.0, 4.0], [5.0, 6.0]]))


def test_goparalleltuple_serial_reduction_sum():
    def func(x):
        return [np.array([x[0]]), np.array([x[1]])]

    result = CC.Settings.GoParallelTuple(func, [[1, 10], [2, 20], [3, 30]], "+")
    assert len(result) == 2, \
        "GoParallelTuple serial reduction returned %d elements, expected 2" % len(result)
    np.testing.assert_array_equal(result[0], np.array([6]))   # 1+2+3
    np.testing.assert_array_equal(result[1], np.array([60]))  # 10+20+30


@pytest.mark.skipif(_has_mpi4py,
                    reason="no-reduction not implemented for mpi4py path")
def test_goparalleltuple_serial_no_reduction():
    def func(x):
        return [np.array([x]), np.array([x * 10])]

    result = CC.Settings.GoParallelTuple(func, [[1], [2]], reduce_op=None)
    assert len(result) == 2, \
        "GoParallelTuple serial no reduction returned %d elements, expected 2" % len(result)



@pytest.mark.skipif(not _has_mpi4py, reason="mpi4py not installed")
def test_goparalleltuple_mpi4py_returns_all_elements():
    """
    Regression test for issue #120.

    GoParallelTuple with mpi4py must return all elements of the function's
    return tuple, not just the first. Even with a single MPI process the
    allgather path is entered and the original bug dropped elements 2+.
    """
    def func(x):
        return [np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])]

    result = CC.Settings.GoParallelTuple(func, [[1, 2]], "+")
    msg = "GoParallelTuple mpi4py returned %d elements, expected 2 (issue #120)" \
          % len(result)
    assert len(result) == 2, msg
    np.testing.assert_array_equal(result[0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result[1], np.array([[3.0, 4.0], [5.0, 6.0]]))


if __name__ == "__main__":
    test_goparalleltuple_serial_returns_all_elements()
    test_goparalleltuple_serial_reduction_sum()
    test_goparalleltuple_serial_no_reduction()
    if _has_mpi4py:
        test_goparalleltuple_mpi4py_returns_all_elements()
    print("ok")
