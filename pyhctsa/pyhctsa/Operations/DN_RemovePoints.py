import numpy as np
from scipy import stats
from PeripheryFunctions.BF_iszscored import BF_iszscored
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.CO_FirstCrossing import CO_FirstCrossing
import warnings

def DN_RemovePoints(y, removeHow = 'absfar', p = 0.1, removeOrSaturate = 'remove'):
    """
    DN_RemovePoints: How time-series properties change as points are removed.

    A proportion, p, of points are removed from the time series according to some
    rule, and a set of statistics are computed before and after the change.

    Parameters:
    y (array-like): The input time series
    remove_how (str): How to remove points from the time series:
                      'absclose': those that are the closest to the mean,
                      'absfar': those that are the furthest from the mean,
                      'min': the lowest values,
                      'max': the highest values,
                      'random': at random.
    p (float): The proportion of points to remove (default: 0.1)
    remove_or_saturate (str): To remove points ('remove') or saturate their values ('saturate')

    Returns:
    dict: Statistics including the change in autocorrelation, time scales, mean, spread, and skewness.
    """
    N = len(y) # time series length

    # check that the input time series has been z-scored
    if not BF_iszscored(y):
        warnings.warn("The input time series should be z-scored.")
    
    if removeHow == 'absclose':
        is_ = np.argsort(np.abs(y))[::-1]
    elif removeHow == 'absfar':
        is_ = np.argsort(np.abs(y))
    elif removeHow == 'min':
        is_ = np.argsort(y)[::-1]
    elif removeHow == 'max':
        is_ = np.argsort(y)
    elif removeHow == 'random':
        is_ = np.random.permutation(N)
    else:
        raise ValueError(f"Unknown method '{removeHow}'")
    
    # Indices of points to *keep*:
    rKeep = np.sort(is_[:round(N * (1 - p))])

    # Indices of points to *transform*:
    rTransform = np.setdiff1d(np.arange(N), rKeep)

    # Do the removing/saturating to convert y -> yTransform
    if removeOrSaturate == 'remove':
        yTransform = y[rKeep]
    elif removeOrSaturate == 'saturate':
        # Saturate out the targeted points
        if removeHow == 'max':
            yTransform = y.copy()
            yTransform[rTransform] = np.max(y[rKeep])
        elif removeHow == 'min':
            yTransform = y.copy()
            yTransform[rTransform] = np.min(y[rKeep])
        elif removeHow == 'absfar':
            yTransform = y.copy()
            yTransform[yTransform > np.max(y[rKeep])] = np.max(y[rKeep])
            yTransform[yTransform < np.min(y[rKeep])] = np.min(y[rKeep])
        else:
            raise ValueError(f"Cannot 'saturate' when using '{removeHow}' method")
    else:
        raise ValueError(f"Unknown removOrSaturate option '{removeOrSaturate}'")
    
    # Compute some autocorrelation properties
    acf_y = SUB_acf(y,8)
    acf_yTransform = SUB_acf(yTransform,8)

    # Compute output statistics
    out = {}

    # Helper functions
    f_absDiff = lambda x1, x2: abs(x1 - x2) # ignores the sign
    f_ratio = lambda x1, x2: x1 / x2 # includes the sign

    out['fzcacrat'] = f_ratio(CO_FirstCrossing(yTransform, 'ac', 0, 'continuous'), 
                              CO_FirstCrossing(y, 'ac', 0, 'continuous'))
    
    out['ac1rat'] = f_ratio(acf_yTransform[0], acf_y[0])
    out['ac1diff'] = f_absDiff(acf_yTransform[0], acf_y[0])

    out['ac2rat'] = f_ratio(acf_yTransform[1], acf_y[1])
    out['ac2diff'] = f_absDiff(acf_yTransform[1], acf_y[1])
    
    out['ac3rat'] = f_ratio(acf_yTransform[2], acf_y[2])
    out['ac3diff'] = f_absDiff(acf_yTransform[2], acf_y[2])
    
    out['sumabsacfdiff'] = np.sum(np.abs(acf_yTransform - acf_y))
    out['mean'] = np.mean(yTransform)
    out['median'] = np.median(yTransform)
    out['std'] = np.std(yTransform, ddof=1)
    
    out['skewnessrat'] = stats.skew(yTransform) / stats.skew(y)
    # return kurtosis instead of excess kurtosis
    out['kurtosisrat'] = stats.kurtosis(yTransform, fisher=False) / stats.kurtosis(y, fisher=False)

    return out

def SUB_acf(x, n):
    # computes autocorrelation of the input sequence, x, up to a maximum time lag, n
    acf = CO_AutoCorr(x, list(range(1, n+1)), 'Fourier')

    return acf
