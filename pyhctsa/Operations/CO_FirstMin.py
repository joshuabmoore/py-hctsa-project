import numpy as np
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.IN_AutoMutualInfo import IN_AutoMutualInfo
from PeripheryFunctions.BF_MutualInformation import BF_MutualInformation
import warnings

def CO_FirstMin(y, minWhat = 'mi-gaussian', extraParam = None, minNotMax = True):
    """
    Time of first minimum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.
    minWhat : str, optional
        The type of correlation to minimize. Options are 'ac' for autocorrelation,
        or 'mi' for automutual information. By default, 'mi' specifies the
        'gaussian' method from the Information Dynamics Toolkit. Other options
        include 'mi-kernel', 'mi-kraskov1', 'mi-kraskov2' (from Information Dynamics Toolkit),
        or 'mi-hist' (histogram-based method). Default is 'mi'.
    extraParam : any, optional
        An additional parameter required for the specified `minWhat` method (e.g., for Kraskov).
    minNotMax : bool, optional
        If True, return the maximum instead of the minimum. Default is False.

    Returns
    -------
    int
        The time of the first minimum (or maximum if `minNotMax` is True).
    """

    N = len(y)

    # Define the autocorrelation function
    if minWhat in ['ac', 'corr']:
        # Autocorrelation implemented as CO_AutoCorr
        corrfn = lambda x : CO_AutoCorr(y, tau=x, method='Fourier')
    elif minWhat == 'mi-hist':
        # if extraParam is none, use default num of bins in BF_MutualInformation (default : 10)
        corrfn = lambda x : BF_MutualInformation(y[:-x], y[x:], 'range', 'range', extraParam or 10)
    elif minWhat == 'mi-kraskov2':
        # (using Information Dynamics Toolkit)
        # extraParam is the number of nearest neighbors
        corrfn = lambda x : IN_AutoMutualInfo(y, x, 'kraskov2', extraParam)
    elif minWhat == 'mi-kraskov1':
        # (using Information Dynamics Toolkit)
        corrfn = lambda x : IN_AutoMutualInfo(y, x, 'kraskov1', extraParam)
    elif minWhat == 'mi-kernel':
        corrfn = lambda x : IN_AutoMutualInfo(y, x, 'kernel', extraParam)
    elif minWhat in ['mi', 'mi-gaussian']:
        corrfn = lambda x : IN_AutoMutualInfo(y, x, 'gaussian', extraParam)
    else:
        raise ValueError(f"Unknown correlation type specified: {minWhat}")
    
    # search for a minimum (incrementally through time lags until a minimum is found)
    autoCorr = np.zeros(N-1) # pre-allocate maximum length autocorrelation vector
    if minNotMax:
        # FIRST LOCAL MINUMUM 
        for i in range(1, N):
            autoCorr[i-1] = corrfn(i)
            # Hit a NaN before got to a minimum -- there is no minimum
            if np.isnan(autoCorr[i-1]):
                warnings.warn(f"No minimum in {minWhat} [[time series too short to find it?]]")
                out = np.nan
            
            # we're at a local minimum
            if (i == 2) and (autoCorr[1] > autoCorr[0]):
                # already increases at lag of 2 from lag of 1: a minimum (since ac(0) is maximal)
                return 1
            elif (i > 2) and autoCorr[i-3] > autoCorr[i-2] < autoCorr[i-1]:
                # minimum at previous i
                return i-1 # I found the first minimum!
    else:
        # FIRST LOCAL MAXIMUM
        for i in range(1, N):
            autoCorr[i-1] = corrfn(i)
            # Hit a NaN before got to a max -- there is no max
            if np.isnan(autoCorr[i-1]):
                warnings.warn(f"No minimum in {minWhat} [[time series too short to find it?]]")
                return np.nan

            # we're at a local maximum
            if i > 2 and autoCorr[i-3] < autoCorr[i-2] > autoCorr[i-1]:
                return i-1

    return N
