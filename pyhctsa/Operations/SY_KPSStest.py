from statsmodels.tsa.stattools import kpss
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
# temporarily turn off warnings for the test statistic being too big or small
warnings.simplefilter('ignore', InterpolationWarning)

def SY_KPSStest(y, lags = 0):
    """
    The KPSS stationarity test.
    
    The KPSS stationarity test, of Kwiatkowski, Phillips, Schmidt, and Shin:
    "Testing the null hypothesis of stationarity against the alternative of a
    unit root: How sure are we that economic time series have a unit root?"
    Kwiatkowski, Denis and Phillips, Peter C. B. and Schmidt, Peter and Shin, Yongcheol
    J. Econometrics, 54(1-3) 159 (2002)
    
    Uses the function kpss from statsmodels. The null
    hypothesis is that a univariate time series is trend stationary, the
    alternative hypothesis is that it is a non-stationary unit-root process.
    
    The code can implemented for a specific time lag, tau. Alternatively, measures
    of change in p-values and test statistics will be outputted if the input is a
    vector of time lags.

    Parameters:
    -----------
    y : array_like
        The time series to analyze.
    lags: int or list, optional
        can be either a scalar (returns basic test statistic and p-value), or
        list (returns statistics on changes across these time lags)
    
    Returns:
    --------
    out : dict 
        Either the basic test statistic and p-value or statistics on 
        changes across specified time lags.

    """
    if isinstance(lags, list):
        # evaluate kpss at multiple lags
        pValue = np.zeros(len(lags))
        stat = np.zeros(len(lags))
        for (i, l) in enumerate(lags):
            s, pv, _, _ = kpss(y, nlags=l, regression='ct')
            pValue[i] = pv
            stat[i] = s
        out = {}
        # return stats on outputs
        out['maxpValue'] = np.max(pValue)
        out['minpValue'] = np.min(pValue)
        out['maxstat'] = np.max(stat)
        out['minstat'] = np.min(stat)
        out['lagmaxstat'] = lags[np.argmax(stat)]
        out['lagminstat'] = lags[np.argmin(stat)]
    else:
        if isinstance(lags, int):
            stat, pValue, _, _ = kpss(y, nlags=lags, regression='ct')
            # return the statistic and pvalue
            out = {'stat': stat, 'pValue': pValue}
        else:
            raise TypeError("Expected either a single lag (as an int) or list of lags.")
    
    return out
