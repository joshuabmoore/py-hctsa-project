import numpy as np
from scipy.stats import gaussian_kde, kurtosis, skew, expon
from loguru import logger
from utilities import point_of_crossing

# def FirstMin(y, minWhat = 'mi-gaussian', extraParam = None, minNotMax = True):
#     """
#     Time of first minimum in a given self-correlation function.

#     Parameters
#     ----------
#     y : array-like
#         The input time series.
#     minWhat : str, optional
#         The type of correlation to minimize. Options are 'ac' for autocorrelation,
#         or 'mi' for automutual information. By default, 'mi' specifies the
#         'gaussian' method from the Information Dynamics Toolkit. Other options
#         include 'mi-kernel', 'mi-kraskov1', 'mi-kraskov2' (from Information Dynamics Toolkit),
#         or 'mi-hist' (histogram-based method). Default is 'mi'.
#     extraParam : any, optional
#         An additional parameter required for the specified `minWhat` method (e.g., for Kraskov).
#     minNotMax : bool, optional
#         If True, return the maximum instead of the minimum. Default is False.

#     Returns
#     -------
#     int
#         The time of the first minimum (or maximum if `minNotMax` is True).
#     """

#     N = len(y)

#     # Define the autocorrelation function
#     if minWhat in ['ac', 'corr']:
#         # Autocorrelation implemented as CO_AutoCorr
#         corrfn = lambda x : AutoCorr(y, tau=x, method='Fourier')
#     elif minWhat == 'mi-hist':
#         # if extraParam is none, use default num of bins in BF_MutualInformation (default : 10)
#         corrfn = lambda x : BF_MutualInformation(y[:-x], y[x:], 'range', 'range', extraParam or 10)
#     elif minWhat == 'mi-kraskov2':
#         # (using Information Dynamics Toolkit)
#         # extraParam is the number of nearest neighbors
#         corrfn = lambda x : IN_AutoMutualInfo(y, x, 'kraskov2', extraParam)
#     elif minWhat == 'mi-kraskov1':
#         # (using Information Dynamics Toolkit)
#         corrfn = lambda x : IN_AutoMutualInfo(y, x, 'kraskov1', extraParam)
#     elif minWhat == 'mi-kernel':
#         corrfn = lambda x : IN_AutoMutualInfo(y, x, 'kernel', extraParam)
#     elif minWhat in ['mi', 'mi-gaussian']:
#         corrfn = lambda x : IN_AutoMutualInfo(y, x, 'gaussian', extraParam)
#     else:
#         raise ValueError(f"Unknown correlation type specified: {minWhat}")
    
#     # search for a minimum (incrementally through time lags until a minimum is found)
#     autoCorr = np.zeros(N-1) # pre-allocate maximum length autocorrelation vector
#     if minNotMax:
#         # FIRST LOCAL MINUMUM 
#         for i in range(1, N):
#             autoCorr[i-1] = corrfn(i)
#             # Hit a NaN before got to a minimum -- there is no minimum
#             if np.isnan(autoCorr[i-1]):
#                 warnings.warn(f"No minimum in {minWhat} [[time series too short to find it?]]")
#                 out = np.nan
            
#             # we're at a local minimum
#             if (i == 2) and (autoCorr[1] > autoCorr[0]):
#                 # already increases at lag of 2 from lag of 1: a minimum (since ac(0) is maximal)
#                 return 1
#             elif (i > 2) and autoCorr[i-3] > autoCorr[i-2] < autoCorr[i-1]:
#                 # minimum at previous i
#                 return i-1 # I found the first minimum!
#     else:
#         # FIRST LOCAL MAXIMUM
#         for i in range(1, N):
#             autoCorr[i-1] = corrfn(i)
#             # Hit a NaN before got to a max -- there is no max
#             if np.isnan(autoCorr[i-1]):
#                 warnings.warn(f"No minimum in {minWhat} [[time series too short to find it?]]")
#                 return np.nan

#             # we're at a local maximum
#             if i > 2 and autoCorr[i-3] < autoCorr[i-2] > autoCorr[i-1]:
#                 return i-1

#     return N


def AutoCorr(y : list, tau : int = 1, method : str = 'Fourier'):
    """
    Compute the autocorrelation of an input time series.

    Parameters:
    -----------
    y : array_like
        A scalar time series column vector.
    tau : int, list, optional
        The time-delay. If tau is a scalar, returns autocorrelation for y at that
        lag. If tau is a list, returns autocorrelations for y at that set of
        lags. If empty list, returns the full function for the 'Fourier' estimation method.
    method : str, optional
        The method of computing the autocorrelation: 'Fourier',
        'TimeDomainStat', or 'TimeDomain'.

    Returns:
    --------
    out : float or array
        The autocorrelation at the given time lag(s).

    Notes:
    ------
    Specifying method = 'TimeDomain' can tolerate NaN values in the time
    series.
    """
    y = np.array(y)
    N = len(y)  # time-series length

    if tau:
        # if list is not empty
        if np.max(tau) > N - 1:  # -1 because acf(1) is lag 0
            logger.warning(f"Time lag {np.max(tau)} is too long for time-series length {N}.")
        if np.any(np.array(tau) < 0):
            logger.warning('Negative time lags not applicable.')
    
    if method == 'Fourier':
        n_fft = 2 ** (int(np.ceil(np.log2(N))) + 1)
        F = np.fft.fft(y - np.mean(y), n_fft)
        F = F * np.conj(F)
        acf = np.fft.ifft(F)  # Wiener–Khinchin
        acf = acf / acf[0]  # Normalize
        acf = np.real(acf)
        acf = acf[:N]
        
        if not tau:  # list empty, return the full function
            out = acf
        else:  # return a specific set of values
            tau = np.atleast_1d(tau)
            out = np.zeros(len(tau))
            for i, t in enumerate(tau):
                if (t > len(acf) - 1) or (t < 0):
                    out[i] = np.nan
                else:
                    out[i] = acf[t]
    
    elif method == 'TimeDomainStat':
        sigma2 = np.std(y, ddof=1)**2  # time-series variance
        mu = np.mean(y)  # time-series mean
        
        def acf_y(t):
            return np.mean((y[:N-t] - mu) * (y[t:] - mu)) / sigma2
        
        tau = np.atleast_1d(tau)
        out = np.array([acf_y(t) for t in tau])
    
    elif method == 'TimeDomain':
        tau = np.atleast_1d(tau)
        out = np.zeros(len(tau))
        
        for i, t in enumerate(tau):
            if np.any(np.isnan(y)):
                good_r = (~np.isnan(y[:N-t])) & (~np.isnan(y[t:]))
                print(f'NaNs in time series, computing for {np.sum(good_r)}/{len(good_r)} pairs of points')
                y1 = y[:N-t]
                y1n = y1[good_r] - np.mean(y1[good_r])
                y2 = y[t:]
                y2n = y2[good_r] - np.mean(y2[good_r])
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean(y1n * y2n) / np.std(y1[good_r], ddof=0) / np.std(y2[good_r], ddof=0)
            else:
                y1 = y[:N-t]
                y2 = y[t:]
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean((y1 - np.mean(y1)) * (y2 - np.mean(y2))) / np.std(y1, ddof=0) / np.std(y2, ddof=0)
    
    else:
        raise ValueError(f"Unknown autocorrelation estimation method '{method}'.")
    
    return out

def FirstCrossing(y : list, corr_fun : str ='ac', threshold : float = 0.0, what_out='both'):
    """
    The first crossing of a given autocorrelation across a given threshold.

    Parameters:
    -----------
    y : array_like
        The input time series
    corr_fun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]
    what_out : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'

    Returns:
    --------
    out : dict or float
        The first crossing information, format depends on what_out
    """
    # Select the self-correlation function
    if corr_fun == 'ac':
        # Autocorrelation at all time lags
        corrs = AutoCorr(y, [], 'Fourier')
    else:
        raise ValueError(f"Unknown correlation function '{corr_fun}'")

    # Calculate point of crossing
    first_crossing_index, point_of_crossing_index = point_of_crossing(corrs, threshold)

    # Assemble the appropriate output (dictionary or float)
    # Convert from index space (1,2,…) to lag space (0,1,2,…)
    if what_out == 'both':
        out = {
            'firstCrossing': first_crossing_index - 1,
            'pointOfCrossing': point_of_crossing_index - 1
        }
    elif what_out == 'discrete':
        out = first_crossing_index - 1
    elif what_out == 'continuous':
        out = point_of_crossing_index - 1
    else:
        raise ValueError(f"Unknown output format '{what_out}'")

    return out
