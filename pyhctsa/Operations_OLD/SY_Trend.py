import numpy as np
from PeripheryFunctions.BF_iszscored import BF_iszscored
import warnings
from scipy.signal import detrend

def SY_Trend(y):
    """
    Quantifies various measures of trend in a time series.

    Linearly detrends the time series using detrend, and returns the ratio of
    standard deviations before and after the linear detrending. If a strong linear
    trend is present in the time series, this operation should output a low value.
    Also fits a line and gives parameters from that fit, as well as statistics on
    a cumulative sum of the time series.

    Parameters:
    -----------
    y : array-like
        the input time series
    
    Returns:
    --------
    out : dict
        a dictionary of various measures of trend in the time series
    """
    if not BF_iszscored(y):
        warnings.warn('The input time series should be z-scored')
    
    N = len(y)

    # ratio of std before and after linear detrending
    out = {}
    dt_y = detrend(y)
    out['stdRatio'] = np.std(dt_y, ddof=1) / np.std(y, ddof=1)
    
    # do a linear fit
    # need to use the same xrange as MATLAB with 1 indexing for correct result
    coeffs = np.polyfit(range(1, N+1), y, 1)
    out['gradient'] = coeffs[0]
    out['intercept'] = coeffs[1]

    # Stats on the cumulative sum
    yC = np.cumsum(y)
    out['meanYC'] = np.mean(yC)
    out['stdYC'] = np.std(yC, ddof=1)
    coeffs_yC = np.polyfit(range(1, N+1), yC, 1)
    out['gradientYC'] = coeffs_yC[0]
    out['interceptYC'] = coeffs_yC[1]

    # Mean cumsum in first and second half of the time series
    out['meanYC12'] = np.mean(yC[:int(np.floor(N/2))])
    out['meanYC22'] = np.mean(yC[int(np.floor(N/2)):])

    return out
