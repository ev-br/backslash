import numpy as np
import scipy.linalg as sl

KNOWN_FORMATS = {"general",}

SOLVE_FUNCS = {"general": sl.solve,}


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


    return solve_func(a._data, b)
    
