import numpy as np

def DN_Unique(x):
    """
    The proportion of the time series that are unique values.

    Parameters:
    x (array-like): the input data vector

    Returns:
    out (float): the proportion of time series that are unique values
    """
    out = len(np.unique(x))/len(x)

    return out
