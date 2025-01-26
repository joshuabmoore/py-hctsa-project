import numpy as np 

def DN_TrimmedMean(y, n=0):
    """
    Mean of the trimmed time series using trimmean.

    Parameters:
    ----------
    y (array-like): the input time series
    n (float): the percentage of highest and lowest values in y to exclude from the mean calculation

    Returns:
    --------
    out (float): the mean of the trimmed time series.
    """
    n *= 0.01
    N = len(y)
    trim = int(np.round(N * n / 2))
    y = np.sort(y)

    out = np.mean(y[trim:N-trim])

    return out
