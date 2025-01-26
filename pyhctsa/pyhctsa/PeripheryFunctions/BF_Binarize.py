import numpy as np

def BF_Binarize(y, binarizeHow='diff'):
    """
    Converts an input vector into a binarized version.

    Parameters:
    -----------
    y : array_like
        The input time series
    binarizeHow : str, optional
        Method to binarize the time series: 'diff', 'mean', 'median', 'iqr'.
    Returns:
    --------
    yBin : array_like
        The binarized time series
    """
    if binarizeHow == 'diff':
        # Binary signal: 1 for stepwise increases, 0 for stepwise decreases
        yBin = stepBinary(np.diff(y))
    
    elif binarizeHow == 'mean':
        # Binary signal: 1 for above mean, 0 for below mean
        yBin = stepBinary(y - np.mean(y))
    
    elif binarizeHow == 'median':
        # Binary signal: 1 for above median, 0 for below median
        yBin = stepBinary(y - np.median(y))
    
    elif binarizeHow == 'iqr':
        # Binary signal: 1 if inside interquartile range, 0 otherwise
        iqr = np.quantile(y,[.25,.75])
        iniqr = np.logical_and(y > iqr[0], y<iqr[1])
        yBin = np.zeros(len(y))
        yBin[iniqr] = 1
    else:
        raise ValueError(f"Unknown binary transformation setting '{binarizeHow}'")

    return yBin

def stepBinary(X):
    # Transform real values to 0 if <=0 and 1 if >0:
    Y = np.zeros(len(X))
    Y[X > 0] = 1

    return Y
