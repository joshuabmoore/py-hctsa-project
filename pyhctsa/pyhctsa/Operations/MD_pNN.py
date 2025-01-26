import numpy as np

def MD_pNN(x):
    """
    MD_pNN: pNNx measures of heart rate variability.

    Applies pNNx measures to time series assumed to represent sequences of
    consecutive RR intervals measured in milliseconds.

    This code is derived from MD_hrv_classic.m because it doesn't make medical
    sense to do PNN on a z-scored time series. But now PSD doesn't make too much sense, 
    so we just evaluate the pNN measures.

    Parameters:
    -----------
    x (array-like): Input time series.

    Returns:
    --------
    dict: A dictionary containing the pNNx measures.
    """

    diffx = np.diff(x)
    N = len(x)

    # Calculate pNNx percentage

    Dx = np.abs(diffx) * 1000 # assume milliseconds as for RR intervals
    pnns = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    out = {} # dict used for output in place of MATLAB struct

    for x in pnns:
        out["pnn" + str(x) ] = sum(Dx > x) / (N-1)

    return out
