from scipy.stats import moment
import numpy as np

def DN_Moments(y, theMom):
    """
    A moment of the distribution of the input time series.
    Normalizes by the standard deviation.

    Parameters:
    y (array-like): the input data vector
    theMom (int): the moment to calculate (a scalar)

    Returns:
    out (float): theMom moment of the distribution of the input time series. 
    """
    out = moment(y, theMom) / np.std(y, ddof=1) # normalized

    return out
