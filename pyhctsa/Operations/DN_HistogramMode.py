import numpy as np
from PeripheryFunctions.BF_SimpleBinner import BF_SimpleBinner

def DN_HistogramMode(y, numBins = 10, doSimple = True):
    """
    Mode of a data vector.
    Measures the mode of the data vector using histograms with a given number
    of bins.

    Parameters:
    -----------
    y : array-like
        the input data vector
    numBins : int, optional
        the number of bins to use in the histogram
    doSimple : bool, optional
        whether to use a simple binning method (linearly spaced bins)

    Returns:
    --------
    out : float
        the mode of the data vector using histograms with numBins bins. 
    """

    if isinstance(numBins, int):
        if doSimple:
            N, binEdges = BF_SimpleBinner(y, numBins)
        else:
            # this gives a different result to MATLAB for the same number of bins 
            # better to use the simple binner (as set by default)
            N, binEdges = np.histogram(y, bins=numBins)
    elif isinstance(numBins, str):
        # NOTE: auto doesn't yield the same number of bins as MATLAB's auto, despite both using the same binning algs. 
        bin_edges = np.histogram_bin_edges(y, bins=numBins)
        N, binEdges = np.histogram(y, bins=bin_edges)
    else:
        raise ValueError("Unknown format for numBins")

    # compute bin centers from bin edges
    binCenters = np.mean([binEdges[:-1], binEdges[1:]], axis=0)

    # mean position of maximums (if multiple)
    out = np.mean(binCenters[N == np.max(N)])

    return out
