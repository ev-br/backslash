import numpy as np
import scipy.linalg as sl
from numpy.testing import assert_allclose


SOLVE_FUNCS = {"general": sl.solve, "triangular": sl.solve_triangular} 


def normalize_format(format):
    if format is None:
        format = "general"
    if format not in SOLVE_FUNCS.keys():
        raise ValueError(f"Unknown format {format}")
    return format


class Array(object):
    def __init__(self, data, format=None, *args, **kwds):
        super().__init__(*args, **kwds)
        self._data = data
        self._format = normalize_format(format)

    @property
    def format(self):
        return self._format

    @property
    def data(self):
        return self._data


def solve(a, b):
    """Solve a linear system a@x = b.

    Parameters
    ----------
    a : Array
        Left-hand side
    b : numpy array, 1D
        Right-hand side

    Returns
    -------
    numpy array : Solution of the linear system.
    """

    b = np.asarray(b)
    try:
        solve_func = SOLVE_FUNCS[a.format]
    except KeyError:
        raise ValueError(f"No suitable solve_functions for {format}")

    if np.allclose(a.data, np.tril(a.data)):  # check if lower triangular
        return solve_func(a.data, b, lower=True)
    else:  # if upper triangular then standart
        return solve_func(a.data, b)
