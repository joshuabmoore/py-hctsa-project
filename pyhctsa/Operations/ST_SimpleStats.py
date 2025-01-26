import numpy as np
from PeripheryFunctions.BF_SignChange import BF_SignChange
from PeripheryFunctions.BF_zscore import BF_zscore
from scipy.signal import detrend

def ST_SimpleStats(x, whatStat):
    """
    Basic statistics about an input time series.

    Parameters:
    -----------
    x : array_like
        the input time series
    whatStat : str
        the statistic to return:
          (i) 'zcross': the proportionof zero-crossings of the time series
                        (z-scored input thus returns mean-crossings)
          (ii) 'maxima': the proportion of the time series that is a local maximum
          (iii) 'minima': the proportion of the time series that is a local minimum
          (iv) 'pmcross': the ratio of the number of times that the (ideally
                          z-scored) time-series crosses +1 (i.e., 1 standard
                          deviation above the mean) to the number of times
                          that it crosses -1 (i.e., 1 standard deviation below
                          the meSan)
          (v) 'zsczcross': the ratio of zero crossings of raw to detrended
                           time series where the raw has zero mean
    
    Returns:
    --------
    out : float
        the statistic.
    """

    N = len(x)

    if whatStat == 'zcross':
        # Proportion of zero-crossings of the time series
        # (% in the case of z-scored input, crosses its mean)
        xch = x[:-1] * x[1:]
        out = np.sum(xch < 0)/N

    elif whatStat == 'maxima':
        # proportion of local maxima in the time series
        dx = np.diff(x)
        out = np.sum((dx[:-1] > 0) & (dx[1:] < 0)) / (N - 1)
    elif whatStat == 'minima':
        # proportion of local minima in the time series
        dx = np.diff(x)
        out = np.sum((dx[:-1] < 0) & (dx[1:] > 0)) / (N-1)
    elif whatStat == 'pmcross':
        # ratio of times cross 1 to -1
        c1sig = np.sum(BF_SignChange(x-1)) # num times cross 1
        c2sig = np.sum(BF_SignChange(x+1)) # num times cross -1
        if c2sig == 0:
            out = np.NaN
        else:
            out = c1sig/c2sig
    elif whatStat == 'zsczcross':
        # ratio of zero crossings of raw to detrended time series
        # where the raw has zero mean
        x = BF_zscore(x)
        xch = x[:-1] * x[1:]
        h1 = np.sum(xch < 0) # num of zscross of raw series
        y = detrend(x)
        ych = y[:-1] * y[1:]
        h2 = np.sum(ych < 0) # % of detrended series
        if h1 == 0:
            out = np.NaN
        else:
            out = h2/h1
    else:
        return ValueError(f"Unknown statistic {whatStat}")
    
    return out
