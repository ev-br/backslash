import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg as sl
from pytest import raises

from backslash import Array, solve


def test1_solve_banded():
    #     [5  2 -1  0  0]       [0]
    #     [1  4  2 -1  0]       [1]
    # a = [0  1  3  2 -1]   b = [2]
    #     [0  0  1  2  2]       [2]
    #     [0  0  0  1  1]       [3]
    # non-zero diagonals: lower - 1, upper - 2
    # banded matrix: [0, 0, -1, -1, -1],
    #                [0, 2, 2, 2, 2],
    #                [5, 4, 3, 2, 1],
    #                [1, 1, 1, 1, 0]

    arr = np.array([[5, 2, -1,  0,  0],
                    [1,  4,  2, -1,  0],
                    [0,  1,  3,  2, -1],
                    [0,  0,  1,  2,  2],
                    [0,  0,  0,  1,  1]], dtype=float)
    arr_banded = np.array([[0, 0, -1, -1, -1],
                           [0, 2, 2, 2, 2],
                           [5, 4, 3, 2, 1],
                           [1, 1, 1, 1, 0]], dtype=float)
    b = np.array([0, 1, 2, 2, 3], dtype=float)
    a = Array(arr, format="banded")
    x1 = solve(a, b)
    x2 = sl.solve_banded((1, 2), arr_banded, b)
    assert_allclose(x1, x2, atol=1e-14)


def test2_solve_banded():
    #     [4  0  0  0]       [0]
    #     [2  3  1  0]       [1]
    # a = [0  1  2  2]   b = [2]
    #     [0  0  0  1]       [2]
    # non-zero diagonals: lower - 1, upper - 1
    # banded matrix: [0, 0, 1, 2],
    #                [4, 3, 2, 1],
    #                [2, 1, 0, 0],
    arr = np.array([[4, 0, 0, 0],
                    [2, 3, 1, 0],
                    [0, 1, 2, 2],
                    [0, 0, 0, 1]], dtype=float)
    arr_banded = np.array([[0, 0, 1, 2],
                           [4, 3, 2, 1],
                           [2, 1, 0, 0]], dtype=float)
    b = np.array([0, 1, 2, 2], dtype=float)
    a = Array(arr, format="banded")
    x1 = solve(a, b)
    x2 = sl.solve_banded((1, 1), arr_banded, b)
    assert_allclose(x1, x2, atol=1e-14)
