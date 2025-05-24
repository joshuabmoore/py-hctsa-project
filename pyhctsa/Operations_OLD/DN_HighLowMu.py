import numpy as np 

def DN_HighLowMu(y):
    """
    The highlowmu statistic.

    The highlowmu statistic is the ratio of the mean of the data that is above the
    (global) mean compared to the mean of the data that is below the global mean.

    Paramters:
    ----------
    y (array-like): The input data vector

    Returns:
    --------
    float: The highlowmu statistic.
    """
    mu = np.mean(y) # mean of data
    mhi = np.mean(y[y > mu]) # mean of data above the mean
    mlo = np.mean(y[y < mu]) # mean of data below the mean
    out = np.divide((mhi-mu), (mu-mlo)) # ratio of the differences

    return out
