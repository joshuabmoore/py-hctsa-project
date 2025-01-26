import numpy as np

def SY_DriftingMean(y, segmentHow = 'num', l = None):
    """
    Mean and variance in local time-series subsegments.
    Splits the time series into segments, computes the mean and variance in each
    segment and compares the maximum and minimum mean to the mean variance.

    This function implements an idea found in the Matlab Central forum:
    http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters:
    -----------
    y : array-like
        the input time series
    segmentHow : str, optional
        (i) 'fix': fixed-length segments (of length l)
        (ii) 'num': a given number, l, of segments
    l : int, optional
        either the length ('fix') or number of segments ('num')
    
    Returns:
    --------
    out : dict
        dictionary of statistics pertaining to mean and variance in local time-series subsegments
    """
    N = len(y)
    
    if l is None:
        if segmentHow == 'num':
            l = 5 # 5 segments
        elif segmentHow == 'fix':
            l = 200 # 200 sample segments

    if segmentHow == 'num':
        l = int(np.floor(N/l))
    elif segmentHow != 'fix':
        raise ValueError(f"Unknown input setting {segmentHow}")
    
    # Check for short time series
    if l == 0 or N < l: # doesn't make sense to split into more windows than there are data points
        return np.NaN
    
    # get going
    numFits = int(np.floor(N/l)) # number of times l fits completely into N
    z = np.zeros((l, numFits))
    for i in range(numFits):
        z[:, i] = y[i*l : (i+1)*l]
    zm = np.mean(z, axis=0)
    zv = np.var(z, ddof=1, axis=0)
    meanVar = np.mean(zv)
    maxMean = np.max(zm)
    minMean = np.min(zm)
    meanMean = np.mean(zm)

    # Output stats
    out = {}
    out['max'] = maxMean/meanVar
    out['min'] = minMean/meanVar
    out['mean'] = meanMean/meanVar
    out['meanmaxmin'] = (out['max'] + out['min'])/2
    out['meanabsmaxmin'] = (np.abs(out['max']) + np.abs(out['min']))/2

    return out
