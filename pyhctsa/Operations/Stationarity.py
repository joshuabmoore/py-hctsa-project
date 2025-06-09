import numpy as np
from Correlation import FirstCrossing, AutoCorr
from numpy.typing import ArrayLike
from typing import Dict, Union
from loguru import logger
from scipy.signal import detrend
from scipy.stats import skew, kurtosis
from utilities import signChange, ZScore
from Information import MutualInfo
from Entropy import SampleEntropy
from statsmodels.tsa.stattools import kpss

def LocalGlobal(y : ArrayLike, subsetHow : str = 'l', nsamps : Union[int, float, None] = None, randomSeed : int = 0) -> dict:
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
    y = np.asarray(y)

    if nsamps is None:
        if subsetHow in ['l', 'unicg', 'randcg']:
            nsamps = 100 # 100 samples
        elif subsetHow == 'p':
            nsamps = 0.1 # 10 % of time series
    
    N = len(y)

    # Determine subset range to use: r
    if subsetHow == 'l':
        # take first n pts of time series
        r = np.arange(min(nsamps, N))
    elif subsetHow == 'p':
        # take initial proportion n of time series
        r = np.arange(int(np.ceil(N*nsamps)))
    elif subsetHow == 'unicg':
        r = np.round(np.linspace(1, N, nsamps)).astype(int) - 1
    elif subsetHow == 'randcg':
        np.random.seed(randomSeed) # set seed for reproducibility
        # Take n random points in time series; there could be repeats
        r = np.random.randint(0, N, nsamps)
    else:
        raise ValueError(f"Unknown specifier, {subsetHow}. Can be either 'l', 'p', 'unicg', or 'randcg'.")

    if len(r) < 5:
        # It's not really appropriate to compute statistics on less than 5 datapoints
        logger.warning(f"Time series (of length {N}) is too short")
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
    out['ac1'] = np.abs(1 - (AutoCorr(y[r], 1, 'Fourier')[0]/AutoCorr(y, 1, 'Fourier')[0]))
    out['sampen101'] = SampleEntropy(y[r], 1, 0.1)['sampen1']/SampleEntropy(y, 1, 0.1)['sampen1']

    return out

def KPSSTest(y : ArrayLike, lags : Union[int, list] = 0) -> dict:
    """
    Performs the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.

    This implementation uses the statsmodels kpss function to test whether a time series
    is trend stationary. The null hypothesis is that the time series is trend stationary,
    while the alternative hypothesis is that it is a non-stationary unit-root process.

    The test was introduced in:
    Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing the null 
    hypothesis of stationarity against the alternative of a unit root: How sure are we 
    that economic time series have a unit root? Journal of Econometrics, 54(1-3), 159-178.

    The function can be used in two ways:
    1. With a single lag value - returns basic test statistic and p-value
    2. With multiple lag values - returns statistics about how the test results 
       change across different lags

    Parameters
    ----------
    y : ArrayLike
        The input time series to analyze for stationarity
    lags : Union[int, list], optional
        Either:
        - A single lag value (int) to compute the test statistic and p-value
        - A list of lag values to analyze how the test results vary across lags
        Default is 0.

    Returns
    -------
    Dict[str, float]
        The KPSS test statistic and p-value of the test.
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

# def DynWin(y : ArrayLike, maxNumSegments : int = 10) -> dict:
#     """
#     How stationarity estimates depend on the number of time-series subsegments.
    
#     Specifically, variation in a range of local measures are implemented: mean,
#     standard deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1),
#     AC(2), and the first zero-crossing of the autocorrelation function.
    
#     The standard deviation of local estimates of these quantities across the time
#     series are calculated as an estimate of the stationarity in this quantity as a
#     function of the number of splits, n_{seg}, of the time series.

#     Parameters:
#     -----------
#     y : array_like
#         the time series to analyze.
#     maxNumSegments : int, optional
#         the maximum number of segments to consider. Sweeps from 2 to
#         maxNumSegments. Defaults to 10. 
    
#     Returns:
#     --------
#     out : dict
#         the standard deviation of this set of 'stationarity' estimates across these window sizes
#     """
#     y = np.asarray(y)
#     nsegr = np.arange(2, maxNumSegments+1, 1) # range of nseg to sweep across
#     nmov = 1 # controls window overlap
#     numFeatures = 11 # num of features
#     fs = np.zeros((len(nsegr), numFeatures)) # standard deviation of feature values over windows
#     taug = FirstCrossing(y, 'ac', 0, 'discrete') # global tau

#     for i, nseg in enumerate(nsegr):
#         wlen = int(np.floor(len(y)/nseg)) # window length
#         inc = int(np.floor(wlen/nmov)) # increment to move at each step
#         # if increment is rounded to zero, prop it up
#         if inc == 0:
#             inc = 1
        
#         numSteps = int(np.floor((len(y) - wlen)/inc) + 1)
#         qs = np.zeros((numSteps, numFeatures))

#         for j in range(numSteps):
#             ySub = y[j*inc:j*inc+wlen]
#             taul = FirstCrossing(ySub, 'ac', 0, 'discrete')

#             qs[j, 0] = np.mean(ySub)
#             qs[j, 1] = np.std(ySub, ddof=1)
#             qs[j, 2] = skew(ySub)
#             qs[j, 3] = kurtosis(ySub)
#             sampenOut = SampleEntropy(ySub, 2, 0.15)
#             qs[j, 4] = sampenOut['quadSampEn1'] # SampEn_1_015
#             qs[j, 5] = sampenOut['quadSampEn2'] # SampEn_2_015
#             qs[j, 6] = AutoCorr(ySub, 1, 'Fourier')[0] # AC1
#             qs[j, 7] = AutoCorr(ySub, 2, 'Fourier')[0] # AC2
#             # (Sometimes taug or taul can be longer than ySub; then these will output NaNs:)
#             qs[j, 8] = AutoCorr(ySub, taug, 'Fourier')[0] # AC_glob_tau
#             qs[j, 9] = AutoCorr(ySub, taul, 'Fourier')[0] # AC_loc_tau
#             qs[j, 10] = taul
        
#         print(qs)
#         fs[i, :numFeatures] = np.std(qs, ddof=1, axis=0)

#     # fs contains std of quantities at all different 'scales' (segment lengths)
#     fs = np.std(fs, ddof=1, axis=0) # how much does the 'std stationarity' vary over different scales?

#     # Outputs
#     out = {}
#     out['stdmean'] = fs[0]
#     out['stdstd'] = fs[1]
#     out['stdskew'] = fs[2]
#     out['stdkurt'] = fs[3]
#     out['stdsampen1_015'] = fs[4]
#     out['stdsampen2_015'] = fs[5]
#     out['stdac1'] = fs[6]
#     out['stdac2'] = fs[7]
#     out['stdactaug'] = fs[8]
#     out['stdactaul'] = fs[9]
#     out['stdtaul'] = fs[10]

#     return out 

def DriftingMean(y: ArrayLike, segmentHow: str = 'fix', l: int = 20) -> Dict[str, float]:
    """
    Measures mean drift by analyzing mean and variance in time-series subsegments.

    This operation splits a time series into segments, computes the mean and variance 
    in each segment, and compares the maximum and minimum means to the mean variance. 
    This helps identify if the time series has a drifting mean by comparing local 
    statistics across different segments.

    The method follows this approach:
    1. Splits signal into frames of length N (or num segments)
    2. Computes means of each frame
    3. Computes variance for each frame
    4. Compares ratio of max/min means with mean variance

    Original idea by Rune from Matlab Central forum:
    http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters
    ----------
    y : ArrayLike
        The input time series
    segmentHow : str, optional
        Method to segment the time series:
        - 'fix': fixed-length segments of length l (default)
        - 'num': splits into l number of segments
    l : int, optional
        Specifies either:
        - The length of segments when segmentHow='fix' (default=20)
        - The number of segments when segmentHow='num'

    Returns
    -------
    Dict[str, float]
        Dictionary containing the measures of mean drift.
    """
    y = np.asarray(y)
    N = len(y)
    
    # Set default segment parameters
    if l is None:
        l = 200 if segmentHow == 'fix' else 5
    
    # Calculate segment length
    if segmentHow == 'num':
        segment_length = int(np.floor(N/l))
    elif segmentHow == 'fix':
        segment_length = l
    else:
        raise ValueError(f"segmentHow must be 'fix' or 'num', got {segmentHow}")
    
    # Validate segment length
    if segment_length <= 1 or segment_length > N:
        return {
            'max': np.nan,
            'min': np.nan,
            'mean': np.nan,
            'meanmaxmin': np.nan,
            'meanabsmaxmin': np.nan
        }
    
    # Calculate number of complete segments
    num_segments = int(np.floor(N/segment_length))
    
    # More efficient segmentation using array operations
    segments = y[:num_segments * segment_length].reshape(num_segments, segment_length)
    
    # Calculate statistics
    segment_means = np.mean(segments, axis=1)
    segment_vars = np.var(segments, axis=1, ddof=1)
    mean_var = np.mean(segment_vars)
    
    # Prepare output statistics
    out = {
        'max': np.max(segment_means) / mean_var,
        'min': np.min(segment_means) / mean_var,
        'mean': np.mean(segment_means) / mean_var
    }
    out['meanmaxmin'] = (out['max'] + out['min']) / 2
    out['meanabsmaxmin'] = (np.abs(out['max']) + np.abs(out['min'])) / 2

    return out

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