import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.EN_SampEn import EN_SampEn
from Operations.CO_AutoCorr import CO_AutoCorr
import warnings
from scipy import stats

def SY_SpreadRandomLocal(y, l = 100, numSegs = 100, randomSeed = 0):
    """
    Bootstrap-based stationarity measure.
    numSegs time-series segments of length l are selected at random from the time
    series and in each segment some statistic is calculated: mean, standard
    deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1), AC(2), and the
    first zero-crossing of the autocorrelation function.
    Outputs summarize how these quantities vary in different local segments of the
    time series.

    Parameters:
    -----------
    y: array-like 
        The input time series
    l: int or str, optional
        the length of local time-series segments to analyze as a positive integer
        Can also be a specified character string:
        (i) 'ac2': twice the first zero-crossing of the autocorrelation function
        (ii) 'ac5': five times the first zero-crossing of the autocorrelation function
    numSegs: 
        the number of randomly-selected local segments to analyze
    randomSeed:
        the input to the random number generator to control reproducibility (defaults to 0)

    Returns:
    --------
    the mean and also the standard deviation of this set of 100 local estimates.

    Note: Function is very slow to compute due to reliance on the EN_SampEn function.
    """
    if isinstance(l, str):
        taug = CO_FirstCrossing(y, 'ac', 0, 'discrete')
        if l == 'ac2':
            l = 2 * taug
        elif l == 'ac5':
            l = 5 * taug
        else:
            raise ValueError(f"Unknown specifier '{l}'")
        
        # Very short l for this sort of time series:
        if l < 5:
            print(f"Warning: This time series has a very short correlation length; "
                  f"Setting l={l} means that changes estimates will be difficult to compare...")

    N = len(y)
    if l > 0.9 * N: # operation is not suitable -- time series is too short
        warnings.warn(f"This time series (N = {N}) is too short to use l = {l}")
        return np.NaN
    
    # numSegs segments, each of length segl data points
    numFeat = 8
    qs = np.zeros((numSegs, numFeat))
    # set the random seed for reproducibility
    np.random.seed(randomSeed)

    for j in range(numSegs):
        ist = np.random.randint(N - l)
        ifh = ist + l
        ySub = y[ist:ifh]

        qs[j, 0] = np.mean(ySub)
        qs[j, 1] = np.std(ySub, ddof=1)
        qs[j, 2] = stats.skew(ySub)
        qs[j, 3] = stats.kurtosis(ySub)
        entropyOut = EN_SampEn(ySub, 1, 0.15)
        qs[j, 4] = entropyOut['quadSampEn1']
        qs[j, 5] = CO_AutoCorr(ySub, 1, 'Fourier')[0]
        qs[j, 6] = CO_AutoCorr(ySub, 2, 'Fourier')[0]
        qs[j, 7] = CO_FirstCrossing(ySub, 'ac', 0, 'continuous') # first zero crossing
    
    fs = np.zeros((numFeat, 2))
    fs[:, 0] = np.nanmean(qs, axis=0)
    fs[:, 1] = np.nanstd(qs, axis=0, ddof=1)

    out = {
        'meanmean': fs[0, 0], 'meanstd': fs[1, 0], 'meanskew': fs[2, 0], 'meankurt': fs[3, 0],
        'meansampen1_015': fs[4, 0], 'meanac1': fs[5, 0], 'meanac2': fs[6, 0], 'meantaul': fs[7, 0],
        'stdmean': fs[0, 1], 'stdstd': fs[1, 1], 'stdskew': fs[2, 1], 'stdkurt': fs[3, 1],
        'stdsampen1_015': fs[4, 1], 'stdac1': fs[5, 1], 'stdac2': fs[6, 1], 'stdtaul': fs[7, 1]
    }

    return out
