import numpy as np

def SY_StatAv(y, whatType='seg', n=5):
    """
    Simple mean-stationarity metric.

    The StatAv measure divides the time series into non-overlapping subsegments,
    calculates the mean in each of these segments and returns the standard deviation
    of this set of means.

    Empirically mean-stationary data would display StatAv approaching to zero.

    Args:
    y (array-like): 
        The input time series
    whatType (str): The type of StatAv to perform:
        'seg': divide the time series into n segments (default)
        'len': divide the time series into segments of length n
    n (int): 
        Either the number of subsegments ('seg') (default : 5) or their length ('len').

    Returns:
    out: float 
        The StatAv statistic
    """
    N = len(y)

    if whatType == 'seg':
        # divide time series into n segments
        p = int(np.floor(N / n))  # integer division, lose the last N mod n data points
        M = np.array([np.mean(y[p*j:p*(j+1)]) for j in range(n)])
    elif whatType == 'len':
        if N > 2*n:
            pn = int(np.floor(N / n))
            M = np.array([np.mean(y[j*n:(j+1)*n]) for j in range(pn)])
        else:
            print(f"This time series (N = {N}) is too short for StatAv({whatType},'{n}')")
            return np.nan
    else:
        raise ValueError(f"Error evaluating StatAv of type '{whatType}', please select either 'seg' or 'len'")

    s = np.std(y, ddof=1)  # should be 1 (for a z-scored time-series input)
    sdav = np.std(M, ddof=1)
    out = sdav / s

    return out 
