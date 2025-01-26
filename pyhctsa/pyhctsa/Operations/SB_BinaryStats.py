import numpy as np
from PeripheryFunctions.BF_Binarize import BF_Binarize

def SB_BinaryStats(y, binaryMethod = 'diff'):
    """
    Statistics on a binary symbolization of the time series.

    Binary symbolization of the time series is a symbolic string of 0s and 1s.
    Provides information about the coarse-grained behavior of the time series.

    Parameters:
    -----------
    y : array_like
        The input time series.
    binary_method : str, optional
        The symbolization rule:
        'diff': by whether incremental differences of the time series are
                positive (1), or negative (0),
        'mean': by whether each point is above (1) or below the mean (0)
        'iqr': by whether the time series is within the interquartile range
               (1), or not (0).

    Returns:
    --------
    dict
        A dictionary containing various statistics on the binary symbolization.
    """
    
    # Binarize the time series
    yBin = BF_Binarize(y, binarizeHow=binaryMethod)
    N = len(yBin)

    # Stationarity of binarised time series
    out = {}
    out['pupstat2'] = np.sum(yBin[N//2:] == 1) / np.sum(yBin[:N//2] == 1)

    # Consecutive strings of ones/zeros (normalized by length)
    diff_y = np.diff(np.where(np.concatenate(([1], yBin, [1])))[0])
    stretch0 = diff_y[diff_y != 1] - 1

    diff_y = np.diff(np.where(np.concatenate(([0], yBin, [0])) == 0)[0])
    stretch1 = diff_y[diff_y != 1] - 1

    # pstretches
    # Number of different stretches as proportion of the time-series length
    out['pstretch1'] = len(stretch1) / N

    if len(stretch0) == 0:
        out['longstretch0'] = 0
        out['longstretch0norm'] = 0
        out['meanstretch0'] = 0
        out['meanstretch0norm'] = 0
        out['stdstretch0'] = np.nan
        out['stdstretch0norm'] = np.nan
    else:
        out['longstretch0'] = np.max(stretch0)
        out['longstretch0norm'] = np.max(stretch0) / N
        out['meanstretch0'] = np.mean(stretch0)
        out['meanstretch0norm'] = np.mean(stretch0) / N
        out['stdstretch0'] = np.std(stretch0, ddof=1)
        out['stdstretch0norm'] = np.std(stretch0, ddof=1) / N

    if len(stretch1) == 0:
        out['longstretch1'] = 0
        out['longstretch1norm'] = 0
        out['meanstretch1'] = 0
        out['meanstretch1norm'] = 0
        out['stdstretch1'] = np.nan
    else:
        out['longstretch1'] = np.max(stretch1)
        out['longstretch1norm'] = np.max(stretch1) / N
        out['meanstretch1'] = np.mean(stretch1)
        out['meanstretch1norm'] = np.mean(stretch1) / N
        out['stdstretch1'] = np.std(stretch1, ddof=1)
        out['stdstretch1norm'] = np.std(stretch1, ddof=1) / N
    
    out['meanstretchdiff'] = (out['meanstretch1'] - out['meanstretch0']) / N
    out['stdstretchdiff'] = (out['stdstretch1'] - out['stdstretch0']) / N

    out['diff21stretch1'] = np.mean(stretch1 == 2) - np.mean(stretch1 == 1)
    out['diff21stretch0'] = np.mean(stretch0 == 2) - np.mean(stretch0 == 1)

    return out 
