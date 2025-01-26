import numpy as np
from PeripheryFunctions.BF_iszscored import BF_iszscored
from Operations.CO_AutoCorr import CO_AutoCorr
from PeripheryFunctions.PN_sampenc import PN_sampenc
from warnings import warn
from scipy.stats import skew, kurtosis

def SY_LocalGlobal(y, subsetHow = 'l', n = None, randomSeed = 0):
    """
    Compare local statistics to global statistics of a time series.

    Parameters:
    -----------
    y : array_like
        The time series to analyze.
    subsetHow : str, optional
        The method to select the local subset of time series:
        'l': the first n points in a time series (default)
        'p': an initial proportion of the full time series
        'unicg': n evenly-spaced points throughout the time series
        'randcg': n randomly-chosen points from the time series (chosen with replacement)
    n : int or float, optional
        The parameter for the method specified by subsetHow.
        Default is 100 samples or 0.1 (10% of time series length) if proportion. 
    random_seed : int, optional
        Seed for random number generator (for 'randcg' option).

    Returns:
    --------
    dict
        A dictionary containing various statistical measures comparing
        the subset to the full time series.
    """
    # check input time series is z-scored
    if not BF_iszscored(y):
        warn(f"The input time series should be z-scored")
    
    if n is None:
        if subsetHow in ['l', 'unicg', 'randcg']:
            n = 100 # 100 samples
        elif subsetHow == 'p':
            n = 0.1 # 10 % of time series
    
    N = len(y)

    # Determine subset range to use: r
    if subsetHow == 'l':
        # take first n pts of time series
        r = np.arange(min(n, N))
    elif subsetHow == 'p':
        # take initial proportion n of time series
        r = np.arange(int(np.ceil(N*n)))
    elif subsetHow == 'unicg':
        r = np.round(np.linspace(1, N, n)).astype(int) - 1
    elif subsetHow == 'randcg':
        np.random.seed(randomSeed) # set seed for reproducibility
        # Take n random points in time series; there could be repeats
        r = np.random.randint(0, N, n)
    else:
        raise ValueError(f"Unknown specifier, {subsetHow}. Can be either 'l', 'p', 'unicg', or 'randcg'.")

    if len(r) < 5:
        # It's not really appropriate to compute statistics on less than 5 datapoints
        warn(f"Time series (of length {N}) is too short")
        return np.NaN
    
    # Compare statistics of this subset to those obtained from the full time series
    out = {}
    out['absmean'] = np.abs(np.mean(y[r])) # Makes sense without normalization if y is z-scored
    out['std'] = np.std(y[r], ddof=1) # Makes sense without normalization if y is z-scored
    out['median'] = np.median(y[r]) # if median is very small then normalization could be very noisy
    raw_iqr_yr = np.percentile(y[r], 75, method='hazen') - np.percentile(y[r], 25, method='hazen')
    raw_iqr_y = np.percentile(y, 75, method='hazen') - np.percentile(y, 25, method='hazen')
    out['iqr'] = np.abs(1 - (raw_iqr_yr/raw_iqr_y))
    out['skewness'] = np.abs(1 - (skew(y[r])/skew(y)))
    # use Pearson definition (normal ==> 3.0)
    out['kurtosis'] = np.abs(1 - (kurtosis(y[r], fisher=False)/kurtosis(y, fisher=False)))
    out['ac1'] = np.abs(1 - (CO_AutoCorr(y[r], 1, 'Fourier')[0]/CO_AutoCorr(y, 1, 'Fourier')[0]))
    out['sampen101'] = PN_sampenc(y[r], 1, 0.1, True)[0][0]/PN_sampenc(y, 1, 0.1, True)[0][0]

    return out
