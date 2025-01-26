import numpy as np
import warnings
from PeripheryFunctions.BF_iszscored import BF_iszscored
from PeripheryFunctions.BF_SimpleBinner import BF_SimpleBinner

def DN_HistogramAsymmetry(y, numBins = 10, doSimple = True):
    """
    Measures of distributional asymmetry
    Measures the asymmetry of the histogram distribution of the input data vector.

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
    out : dict
        dictionary containing measures of the asymmetry of the histogram distribution

    """
    if not BF_iszscored(y):
        warnings.warn("DN_HistogramAsymmetry assumes a z-scored (or standardised) input")
    
    # compute the histogram seperately from positive and negative values in the data
    yPos = y[y > 0] # filter out the positive vals
    yNeg = y[y < 0]

    if doSimple:
        countsPos, binEdgesPos = BF_SimpleBinner(yPos, numBins)
        countsNeg, binEdgesNeg = BF_SimpleBinner(yNeg, numBins)
    else:
        countsPos, binEdgesPos = np.histogram(yPos, numBins)
        countsNeg, binEdgesNeg = np.histogram(yNeg, numBins)
    
    # normalise by the total counts
    NnonZero = np.sum(y!=0)
    pPos = countsPos/NnonZero
    pNeg = countsNeg/NnonZero

    # compute bin centers from bin edges
    binCentersPos = np.mean([binEdgesPos[:-1], binEdgesPos[1:]], axis=0)
    binCentersNeg = np.mean([binEdgesNeg[:-1], binEdgesNeg[1:]], axis=0)

    # Histogram counts and overall density differences
    out = {}
    out['densityDiff'] = np.sum(y > 0) - np.sum(y < 0)  # measure of asymmetry about the mean
    out['modeProbPos'] = np.max(pPos)
    out['modeProbNeg'] = np.max(pNeg)
    out['modeDiff'] = out['modeProbPos'] - out['modeProbNeg']

    # Mean position of maximums (if multiple)
    out['posMode'] = np.mean(binCentersPos[pPos == out['modeProbPos']])
    out['negMode'] = np.mean(binCentersNeg[pNeg == out['modeProbNeg']])
    out['modeAsymmetry'] = out['posMode'] + out['negMode']

    return out
