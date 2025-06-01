import numpy as np
from typing import Union
from pyhctsa.Toolboxes.Max_Little.fastdfa_wrapper import fastdfa

def FastDFA(y: Union[list, np.ndarray]) -> float:
    """
    Measures the scaling exponent of the time series using a fast implementation
    of detrended fluctuation analysis (DFA).

    This is a Python wrapper for Max Little's ML_fastdfa code.
    The original fastdfa code is by Max A. Little and publicly available at:
    http://www.maxlittle.net/software/index.php

    Parameters
    ----------
    y : array-like
        Input time series (1D array), fed straight into the fastdfa script.

    Returns
    -------
    alpha : float
        Estimated scaling exponent from log-log linear fit of fluctuation vs interval.
    """
    y = np.asarray(y)
    intervals, flucts = fastdfa(y)
    idx = np.argsort(intervals)
    intervals_sorted = intervals[idx]
    flucts_sorted = flucts[idx]

    # Log-log linear fit
    coeffs = np.polyfit(np.log10(intervals_sorted), np.log10(flucts_sorted), 1)
    alpha = coeffs[0]
    return alpha
