import numpy as np
from Correlation import FirstCrossing
from numpy.typing import ArrayLike
from typing import Dict, Union
from scipy.signal import detrend
from utilities import signChange, ZScore
from Information import MutualInfo

def FitPolynomial(y : ArrayLike, k : int = 1) -> float:
    """
    Goodness of a polynomial fit to a time series

    Usually kind of a stupid thing to do with a time series, but it's sometimes
    somehow informative for time series with large trends.

    Parameters:
    -----------
    y : array_like
        the time series to analyze.
    k : int, optional
        the order of the polynomial to fit to y.

    Returns:
    --------
    out : float
        RMS error of the fit
    """
    N = len(y)
    t = np.arange(1, N + 1)

    # Fit a polynomial to the time series
    cf = np.polyfit(t, y, k)
    f = np.polyval(cf, t) # evaluate the fitted poly
    out = np.mean((y - f)**2) # mean RMS error of fit

    return out

def TSLength(y : ArrayLike) -> int:
    """
    Length of an input data vector.

    Parameters
    -----------
    y : array_like
        the time series to analyze.

    Returns
    --------
    int
        the length of the time series
    """
    return len(np.asarray(y))

# def MomentCorr(x : ArrayLike, windowLength : Union[None, float] = None, wOverlap : Union[None, float] = None, mom1 : str = 'mean', mom2 : str = 'std', whatTransform : str = 'none'):
#     """
#     Correlations between simple statistics in local windows of a time series.
#     The idea to implement this was that of Prof. Nick S. Jones (Imperial College London).

#     Paramters:
#     ----------
#     x : array_like
#         the input time series
#     windowLength : float, optional
#         the sliding window length (can be a fraction to specify or a proportion of the time-series length)
#     wOverlap : 
#         the overlap between consecutive windows as a fraction of the window length
#     mom1, mom2 : str, optional
#         the statistics to investigate correlations between (in each window):
#             (i) 'iqr': interquartile range
#             (ii) 'median': median
#             (iii) 'std': standard deviation (about the local mean)
#             (iv) 'mean': mean
#     whatTransform : str, optional
#         the pre-processing whatTransformormation to apply to the time series before
#         analyzing it:
#            (i) 'abs': takes absolute values of all data points
#            (ii) 'sqrt': takes the square root of absolute values of all data points
#            (iii) 'sq': takes the square of every data point
#            (iv) 'none': does no whatTransformormation
    
#     Returns:
#     --------
#     out : dict
#         dictionary of statistics related to the correlation between simple statistics in local windows of the input time series. 
#     """
#     x = np.asarray(x)
#     N = len(x) # length of the time series

#     if windowLength is None:
#         windowLength = 0.02 # 2% of the time-series length
    
#     if windowLength < 1:
#         windowLength = int(np.ceil(N * windowLength))
    
#     # sliding window overlap length
#     if wOverlap is None:
#         wOverlap = 1/5
    
#     if wOverlap < 1:
#         wOverlap = int(np.floor(windowLength * wOverlap))

#     # Apply the specified whatTransformation
#     if whatTransform == 'abs':
#         x = np.abs(x)
#     elif whatTransform == 'sq':
#         x = x**2
#     elif whatTransform == 'sqrt':
#         x = np.sqrt(np.abs(x))
#     elif whatTransform == 'none':
#         pass
#     else:
#         raise ValueError(f"Unknown transformation {whatTransform}")
    
#     # create the windows
#     x_buff = _buffer(x, windowLength, wOverlap)
#     numWindows = (N/(windowLength - wOverlap)) # number of windows

#     if np.size(x_buff, 1) > numWindows:
#         x_buff = x_buff[:, :-1] # lose the last point

#     pointsPerWindow = np.size(x_buff, 0)
#     if pointsPerWindow == 1:
#         raise ValueError(f"This time series (N = {N}) is too short to extract {numWindows}")
    
#     # okay now we have the sliding window ('buffered') signal, x_buff
#     # first calculate the first moment in all the windows
#     M1 = __calc_me_moments(x_buff, mom1)
#     M2 = __calc_me_moments(x_buff, mom2)
#     print(M1)

#     out = {}
#     rmat = np.corrcoef(M1, M2)
#     R = rmat[0, 1] # correlation coeff
#     out['R'] = R
#     out['absR'] = np.abs(rmat[0, 1])
#     out['density'] = np.ptp(M1) * np.ptp(M2) / N
#     out['mi'] = MutualInfo(M1, M2, 'gaussian')

#     return out

def __calc_me_moments(x_buff, momType):
    # helper function for MomentCorr
    if momType == 'mean':
        moms = np.mean(x_buff, axis=0)
    elif momType == 'std':
        moms = np.std(x_buff, axis=0, ddof=1)
    elif momType == 'median':
        moms = np.median(x_buff, axis=0)
    elif momType == 'iqr':
        moms = np.percentile(x_buff, 75, method='hazen', axis=0) - np.percentile(x_buff, 25, method='hazen', axis=0)
    else:
        raise ValueError(f"Unknown statistic {momType}")
    return moms

def SimpleStats(x : ArrayLike, whatStat : str = 'zcross') -> dict:
    """
    Basic statistics about an input time series.

    This function computes various statistical measures about zero-crossings and local 
    extrema in a time series. This is part of the Stationarity operations from hctsa,
    implementing ST_SimpleStats.

    Parameters
    ----------
    x : array-like
        The input time series
    whatStat : str, optional
        The statistic to return (default is 'zcross'):
        - 'zcross': proportion of zero-crossings (for z-scored input, returns mean-crossings)
        - 'maxima': proportion of points that are local maxima
        - 'minima': proportion of points that are local minima
        - 'pmcross': ratio of crossings above +1σ to crossings below -1σ
        - 'zsczcross': ratio of zero-crossings in raw vs detrended time series

    Returns
    -------
    float
        The calculated statistic based on whatStat
    """
    x = np.asarray(x)
    N = len(x)

    out = None
    if whatStat == 'zcross':
        # Proportion of zero-crossings of the time series
        # (% in the case of z-scored input, crosses its mean)
        xch = x[:-1] * x[1:]
        out = np.sum(xch < 0)/N

    elif whatStat == 'maxima':
        # proportion of local maxima in the time series
        dx = np.diff(x)
        out = np.sum((dx[:-1] > 0) & (dx[1:] < 0)) / (N - 1)
    elif whatStat == 'minima':
        # proportion of local minima in the time series
        dx = np.diff(x)
        out = np.sum((dx[:-1] < 0) & (dx[1:] > 0)) / (N-1)
    elif whatStat == 'pmcross':
        # ratio of times cross 1 to -1
        c1sig = np.sum(signChange(x-1)) # num times cross 1
        c2sig = np.sum(signChange(x+1)) # num times cross -1
        if c2sig == 0:
            out = np.nan
        else:
            out = c1sig/c2sig
    elif whatStat == 'zsczcross':
        # ratio of zero crossings of raw to detrended time series
        # where the raw has zero mean
        x = ZScore(x)
        xch = x[:-1] * x[1:]
        h1 = np.sum(xch < 0) # num of zscross of raw series
        y = detrend(x)
        ych = y[:-1] * y[1:]
        h2 = np.sum(ych < 0) # % of detrended series
        if h1 == 0:
            out = np.NaN
        else:
            out = h2/h1
    else:
        raise(ValueError(f"Unknown statistic {whatStat}"))
    
    return out

def LocalExtrema(y : ArrayLike, howToWindow : str = 'l', n : Union[int, None] = None) -> dict:
    """
    How local maximums and minimums vary across the time series.

    Finds maximums and minimums within given segments of the time series and
    analyzes the results.

    Parameters
    ----------
    y : array-like
        The input time series
    howToWindow : str, optional
        Method to determine window size (default is 'l'):
        - 'l': windows of a given length (n specifies the window length)
        - 'n': specified number of windows to break the time series into (n specifies number of windows)
        - 'tau': sets window length equal to correlation length (first zero-crossing of autocorrelation)
    n : int, optional
        Specifies either:
        - Window length when howToWindow='l' (defaults to 100)
        - Number of windows when howToWindow='n' (defaults to 5)
        - Not used when howToWindow='tau'

    Returns
    -------
    dict
        Statistics about local extrema including:
        - meanrat: ratio of mean maxima to mean absolute minima
        - medianrat: ratio of median maxima to median absolute minima
        - minmax: minimum of local maxima
        - minabsmin: minimum of absolute local minima
        - And other statistical measures of the extrema

    """
    y = np.asarray(y)
    if n is None:
        if howToWindow == 'l':
            n = 100 # 100 sample windows
        elif howToWindow == 'n':
            n = 5 # 5 windows
    
    N = len(y)

    # Set the window length
    if howToWindow == 'l':
        windowLength = n # window length
    elif howToWindow == 'n':
        windowLength = int(np.floor(N/n))
    elif howToWindow == 'tau':
        windowLength = FirstCrossing(y, 'ac', 0, 'discrete')
    else:
        raise ValueError(f"Unknown method {howToWindow}")
    
    if (windowLength > N) or (windowLength <= 1):
        # This feature is unsuitable if the window length exceeds ts
        out = np.NaN
    
    # Buffer the time series
    y_buff = _buffer(y, windowLength) # no overlap
    # each column is a window of samples
    if y_buff[-1, -1] == 0:
        y_buff = y_buff[:, :-1]  # remove last window if zero-padded
    
    numWindows = np.size(y_buff, 1) # number of windows
    print(numWindows)
    # Find local extrema
    locMax = np.max(y_buff, axis=0) # summary of local maxima
    locMin = np.min(y_buff, axis=0) # summary of local minima
    absLocMin = np.abs(locMin) # abs val of local minima
    exti = np.where(absLocMin > locMax)
    loc_ext = locMax.copy()
    loc_ext[exti] = locMin[exti] # local extrema (furthest from mean; either maxs or mins)
    abs_loc_ext = np.abs(loc_ext) # the magnitude of the most extreme events in each window

    # Return Outputs
    out = {
        'meanrat': np.mean(locMax) / np.mean(absLocMin),
        'medianrat': np.median(locMax) / np.median(absLocMin),
        'minmax': np.min(locMax),
        'minabsmin': np.min(absLocMin),
        'minmaxonminabsmin': np.min(locMax) / np.min(absLocMin),
        'meanmax': np.mean(locMax),
        'meanabsmin': np.mean(absLocMin),
        'meanext': np.mean(loc_ext),
        'medianmax': np.median(locMax),
        'medianabsmin': np.median(absLocMin),
        'medianext': np.median(loc_ext),
        'stdmax': np.std(locMax, ddof=1),
        'stdmin': np.std(locMin, ddof=1),
        'stdext': np.std(loc_ext, ddof=1),
        'zcext': np.sum((loc_ext[:-1] * loc_ext[1:]) < 0) / numWindows,
        'meanabsext': np.mean(abs_loc_ext),
        'medianabsext': np.median(abs_loc_ext),
        'diffmaxabsmin': np.sum(np.abs(locMax - absLocMin)) / numWindows,
        'uord': np.sum(np.sign(loc_ext)) / numWindows,
        'maxmaxmed': np.max(locMax) / np.median(locMax),
        'minminmed': np.min(locMin) / np.median(locMin),
        'maxabsext': np.max(abs_loc_ext) / np.median(abs_loc_ext)
    }
    return out

def _buffer(X, n, p=0, opt=None):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html.
    Taken from: https://stackoverflow.com/questions/38453249/does-numpy-have-a-function-equivalent-to-matlabs-buffer 

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from X
    '''
    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(X):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = X[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), X[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = X[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result