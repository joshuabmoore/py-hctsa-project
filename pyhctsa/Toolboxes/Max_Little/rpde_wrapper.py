import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import sys

if sys.platform.startswith("darwin"):
    lib_ext = ".dylib"
elif sys.platform.startswith("win"):
    lib_ext = ".dll"
else:
    lib_ext = ".so"

_lib_path = os.path.join(os.path.dirname(__file__), f"ML_close_ret{lib_ext}")
_lib = ctypes.CDLL(_lib_path)

# Set argument and return types
_lib.close_ret.argtypes = [
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # x (input array)
    ctypes.c_ulong, # vectorElements (length of x)
    ctypes.c_ulong,    # embedDims
    ctypes.c_ulong, # embedDelay
    ctypes.c_double,  # eta
    ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),   # closeRets (output array)
]
_lib.close_ret.restype = None
def close_ret(x, embedDims, embedDelay, eta):
    """
    Python wrapper for the close_ret C function.

    Parameters
    ----------
    x : array-like
        Input time series (1D array).
    embedDims : int
        Embedding dimension.
    embedDelay : int
        Embedding delay.
    eta : float
        Close return distance.

    Returns
    -------
    closeRets : np.ndarray
        Close return time histogram (length = embedElements).
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    vectorElements = x.shape[0]
    embedElements = vectorElements - ((embedDims - 1) * embedDelay)
    if embedElements <= 0:
        raise ValueError("Input too short for given embedding parameters.")
    closeRets = np.zeros(embedElements, dtype=np.uint64)
    _lib.close_ret(x, vectorElements, embedDims, embedDelay, eta, closeRets)
    return closeRets
