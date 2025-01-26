import numpy as np

def BF_iszscored(x):
    """
    Crude check for whether a data vector is z-scored.

    Parameters:
    -----------
        x (array-like): The input time series (or any vector).

    Returns:
    --------
        iszscored (bool): a bool with the verdict.
    """
    numericThreshold = 100*np.finfo(float).eps

    iszscored = ((np.absolute(np.mean(x)) < numericThreshold) & (np.absolute(np.std(x)-1) < numericThreshold))

    return iszscored
