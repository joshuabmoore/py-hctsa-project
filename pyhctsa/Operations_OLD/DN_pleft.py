import numpy as np

def DN_pleft(y, th = 0.1):
    """
    Distance from the mean at which a given proportion of data are more distant.
    
    Measures the maximum distance from the mean at which a given fixed proportion, `th`, of the time-series data points are further.
    Normalizes by the standard deviation of the time series.
    
    Parameters
    ----------
    y : array_like
        The input data vector.
    th : float, optional
        The proportion of data further than `th` from the mean (default is 0.1).
    
    Returns
    -------
    float
        The distance from the mean normalized by the standard deviation.
    
    """
    p = np.quantile(np.abs(y - np.mean(y)), 1-th, method='hazen')

    # A proportion, th, of the data lie further than p from the mean
    out = p/np.std(y, ddof=1)

    return out
