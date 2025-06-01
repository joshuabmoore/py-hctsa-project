import os
import sys
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Determine the correct library extension for the platform
if sys.platform.startswith("darwin"):
    lib_ext = ".dylib"
elif sys.platform.startswith("win"):
    lib_ext = ".dll"
else:
    lib_ext = ".so"

_lib_path = os.path.join(os.path.dirname(__file__), f"MS_nearest{lib_ext}")
_lib = ctypes.CDLL(_lib_path)

# _lib.ms_nearest_entry.restype = None
_lib.nearest.argtypes = [
        ndpointer(ctypes.c_double, flags='F_CONTIGUOUS'),   # x
        ndpointer(ctypes.c_int,    flags='C_CONTIGUOUS'),   # ind
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),   # avect
        ctypes.c_int, ctypes.c_int, ctypes.c_int ]              # tau,m,n

def ms_nearest(x, tau, avect):
    """
    Python wrapper for ms_nearest_entry C function.

    Parameters
    ----------
    x : np.ndarray
        2D array (m x n), columns are vectors to compare (column-major order).
    tau : int
        Exclusion zone.
    avect : np.ndarray
        Normalization vector (length m).

    Returns
    -------
    ind : np.ndarray
        Indices (0-based) of the nearest columns for each column.
    """
    x = np.array(x).astype(np.double, order='F')
    avect = np.ascontiguousarray(avect, dtype=np.float64)
    m, n = x.shape
    ind = np.zeros(n, dtype=np.int32)
    _lib.nearest(x, ind, avect, tau, m, n)
    return ind
