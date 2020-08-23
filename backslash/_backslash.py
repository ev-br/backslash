import numpy as np
import scipy.linalg as sl

KNOWN_FORMATS = {"general", "banded"}
SOLVE_FUNCS = {"general": sl.solve, "banded": sl.solve_banded}


def normalize_format(format):
    if format is None:
        format = "general"
    if format not in KNOWN_FORMATS:
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
    if a.format == "banded":
        diagonals, banded_matrix = find_banded_args(a._data)
        return solve_func(diagonals, banded_matrix, b)
    return solve_func(a._data, b)


def find_banded_args(arr):
    """Find the arguments of scipy.linalg.solve_banded.

    Parameters
    ----------
    arr : numpy array, 2D

    Returns
    -------
    (integer, integer) : Number of non-zero lower and upper diagonals
    numpy array : Banded matrix
    """
    size = len(arr)

    # find the number of lower non-zero diagonals
    lower_number = 0
    for i in range(1, size):
        flag = False
        for j in range(i):
            if arr[size - i + j, j] != 0:
                flag = True
                break
        if flag:
            lower_number = size - i
            break

    # find the number of upper non-zero diagonals
    upper_number = 0
    for i in range(1, size):
        flag = False
        for j in range(i):
            if arr[j, size - i + j] != 0:
                flag = True
                break
        if flag:
            upper_number = size - i
            break
    banded_matrix = np.zeros((lower_number + upper_number + 1, size))

    # find upper diagonals
    for i in range(size - upper_number, size):
        for j in range(i):
            banded_matrix[i + upper_number - size, size - i + j] = arr[j, size - i + j]
    # find main and lower diagonals
    for i in range(size, size - lower_number - 1, -1):
        for j in range(i):
            banded_matrix[size - i + upper_number, j] = arr[size - i + j, j]
    return (lower_number, upper_number), banded_matrix

