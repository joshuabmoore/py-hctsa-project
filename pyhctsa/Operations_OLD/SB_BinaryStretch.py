import numpy as np
from PeripheryFunctions.BF_SignChange import BF_SignChange


def SB_BinaryStretch(x, stretchWhat = 'lseq1'):
    """
    SB_BinaryStretch  Characterizes stretches of 0/1 in time-series binarization.

    (DOESN'T ACTUALLY, see note) measure the longest stretch of consecutive zeros
    or ones in a symbolized time series as a proportion of the time-series length.

    The time series is symbolized to a binary string by whether it's above (1) or
    below (0) its mean.

    Parameters
    ----------
    x : array_like
        The input time series.
    stretch_what : str, optional
        Specifies which stretch to measure (default is 'lseq1'):
        - 'lseq1': Measures something related to consecutive 1s.
        - 'lseq0': Measures something related to consecutive 0s.

    Returns
    -------
    float
        A measure of the longest stretch of consecutive 0s or 1s as a proportion 
        of the time-series length.
    """

    N = len(x) # time series length
    x = np.array(x)
    x = np.where(x > 0, 1, 0)

    if stretchWhat == 'lseq1':
        # longest stretch of 1s [this code doesn't actualy measure this!]
        indices = np.where(x == 1)[0]
        diffs = np.diff(indices) - 1.5
        sign_changes = BF_SignChange(diffs, 1)
        out = np.max(np.diff(sign_changes)) / N
    elif stretchWhat == 'lseq0':
        # longest stretch of 0s [this code doesn't actualy measure this!]
        indices = np.where(x == 0)[0]
        diffs = np.diff(indices) - 1.5
        sign_changes = BF_SignChange(diffs, 1)
        out = np.max(np.diff(sign_changes)) / N
    else:
        raise ValueError(f"Unknown input {stretchWhat}")
    
    return out if out is not None else 0
