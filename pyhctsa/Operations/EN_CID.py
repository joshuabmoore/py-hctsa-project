import numpy as np

def EN_CID(y):
    """
    Simple complexity measure of a time series.

    Estimates of 'complexity' of a time series as the stretched-out length of the
    lines resulting from a line-graph of the time series.

    Parameters:
    y (array-like): the input time series

    Returns:
    out (dict): dictionary of estimates.
    """
    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    minCE1 = f_CE1(np.sort(y))
    minCE2 = f_CE2(np.sort(y))

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':CE1,'CE2':CE2,'minCE1':minCE1,'minCE2':minCE2,
            'CE1_norm':CE1_norm,'CE2_norm':CE2_norm}

    return out

def f_CE1(y):
    # Original definition (in Table 2 of paper cited above)
    # sum -> mean to deal with non-equal time-series lengths
    # (now scales properly with length)
    return np.sqrt(np.mean(np.power(np.diff(y),2)))

def f_CE2(y):
    # Definition corresponding to the line segment example in Fig. 9 of the paper
    # cited above (using Pythagoras's theorum):
    return np.mean(np.sqrt(1 + np.power(np.diff(y),2)))
