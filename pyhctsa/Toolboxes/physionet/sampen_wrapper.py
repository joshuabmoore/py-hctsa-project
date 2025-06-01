import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys

if sys.platform.startswith("darwin"):
    lib_ext = ".dylib"
elif sys.platform.startswith("win"):
    lib_ext = ".dll"
else:
    lib_ext = ".so"

_lib_path = os.path.join(os.path.dirname(__file__), f"libsampen{lib_ext}")
_lib = ctypes.CDLL(_lib_path)

_lib.sampen.argtypes = [
    ndpointer(dtype=ctypes.c_double),  # input array
    ctypes.c_int,                      # m (embedding dimension)
    ctypes.c_double,                   # r (threshold)
    ctypes.c_int,                      # n (length of the time series)
    ndpointer(dtype=ctypes.c_double),  # output array for entropy values
]
_lib.sampen.restype = None

def calculate_sampen(data: np.ndarray, M: int, r: float) -> np.ndarray:
    """Low-level wrapper for C sampen function. Returns array of SampEn for m=0..M."""
    y = np.asarray(data, dtype=np.float64)
    n = len(y)
    result = np.zeros(M + 1, dtype=np.float64)
    _lib.sampen(y, M, r, n, result)
    return result
