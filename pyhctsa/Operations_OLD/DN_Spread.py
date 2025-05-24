import numpy as np
from scipy import stats

def DN_Spread(y, spreadMeasure='std'):
    """
    Measure of spread of the input time series.
    Returns the spread of the raw data vector, as the standard deviation,
    inter-quartile range, mean absolute deviation, or median absolute deviation.
    """
    if spreadMeasure == 'std':
        out = np.std(y, ddof=1)
    elif spreadMeasure == 'iqr':
        out = stats.iqr(y)
    elif spreadMeasure == 'mad':
        out = mad(y)
    elif spreadMeasure == 'mead':
        out = mead(y)
    else:
        raise ValueError('spreadMeasure must be one of std, iqr, mad or mead')

    return out

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def mead(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)
