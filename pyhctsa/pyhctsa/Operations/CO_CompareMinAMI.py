import numpy as np
from Operations.CO_HistogramAMI import CO_HistogramAMI
from scipy import stats
from PeripheryFunctions.BF_SignChange import BF_SignChange

def CO_CompareMinAMI(y, binMethod, numBins = 10):
    """
    Variability in first minimum of automutual information.

    Finds the first minimum of the automutual information by various different
    estimation methods, and sees how this varies over different coarse-grainings
    of the time series.

    Args:
    y (array-like): The input time series
    binMethod (str): The method for estimating mutual information (input to CO_HistogramAMI)
    numBins (int or array-like): The number of bins for the AMI estimation to compare over

    Returns:
    dict: A dictionary containing various statistics on the set of first minimums 
          of the automutual information function
    """
    N = len(y)
    # Range of time lags to consider
    tauRange = np.arange(0, int(np.ceil(N/2))+1)
    numTaus = len(tauRange)

    # range of bin numbers to consider
    if isinstance(numBins, int):
        numBins = [numBins]
    
    numBinsRange = len(numBins)
    amiMins = np.zeros(numBinsRange)

    # Calculate automutual information
    for i in range(numBinsRange):  # vary over number of bins in histogram
        amis = np.zeros(numTaus)
        for j in range(numTaus):  # vary over time lags, tau
            amis[j] = CO_HistogramAMI(y, tauRange[j], binMethod, numBins[i])
            if (j > 1) and ((amis[j] - amis[j-1]) * (amis[j-1] - amis[j-2]) < 0):
                amiMins[i] = tauRange[j-1]
                break
        if amiMins[i] == 0:
            amiMins[i] = tauRange[-1]
    # basic statistics
    out = {}
    out['min'] = np.min(amiMins)
    out['max'] = np.max(amiMins)
    out['range'] = np.ptp(amiMins)
    out['median'] = np.median(amiMins)
    out['mean'] = np.mean(amiMins)
    out['std'] = np.std(amiMins, ddof=1)
    out['nunique'] = len(np.unique(amiMins))
    out['mode'], out['modef'] = stats.mode(amiMins)
    out['modef'] = out['modef']/numBinsRange

    # converged value? 
    out['conv4'] = np.mean(amiMins[-5:])

    # look for peaks (local maxima)
    # % local maxima above 1*std from mean
    # inspired by curious result of periodic maxima for periodic signal with
    # bin size... ('quantiles', [2:80])
    diff_ami_mins = np.diff(amiMins[:-1])
    positive_diff_indices = np.where(diff_ami_mins > 0)[0]
    sign_change_indices = BF_SignChange(diff_ami_mins, 1)

    # Find the intersection of positive_diff_indices and sign_change_indices
    loc_extr = np.intersect1d(positive_diff_indices, sign_change_indices) + 1
    above_threshold_indices = np.where(amiMins > out['mean'] + out['std'])[0]
    big_loc_extr = np.intersect1d(above_threshold_indices, loc_extr)

    # Count the number of elements in big_loc_extr
    out['nlocmax'] = len(big_loc_extr)

    return out
