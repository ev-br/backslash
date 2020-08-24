import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg as sl
from pytest import raises
from backslash import Array, solve


def test_ctor():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    with raises(TypeError):
        Array(arr, meaning=42)


def test_formats():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    # unknown formats raise
    with raises(ValueError):
        Array(arr, format='oops')
    # No format is == 'general'
    a1 = Array(arr)
    a2 = Array(arr, format="general")
    assert_allclose(a1._data, a2._data, atol=1e-14)


def test_format_setter():
    # format attribute is not writeable
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    a = Array(arr)
    with raises(AttributeError):
        a.format = 'gobbledeegook'



def test_solve():
    arr = np.array([[9, 0, 0],
                    [4, 9, 0],
                    [7, 8, 9]], dtype=float)
    b = np.array([1, 1, 1], dtype=float)
    a = Array(arr, format="triangular")
    x1 = solve(a, b)
    x2 = sl.solve(arr, b)
    assert_allclose(x1, x2, atol=1e-14)

    arr = np.array([[6, 8, 7],
                    [0, 5, 4],
                    [0, 0, 4]], dtype=float)
    b = np.array([1, 1, 1], dtype=float)
    a = Array(arr, format="triangular")
    x1 = solve(a, b)
    x2 = sl.solve(arr, b)
    assert_allclose(x1, x2, atol=1e-14)

    arr = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 9]], dtype=float)
    b = np.array([1, 1, 1], dtype=float)
    a = Array(arr, format="triangular")
    x1 = solve(a, b)
    x2 = sl.solve(arr, b)
    assert_allclose(x1, x2, atol=1e-14)
