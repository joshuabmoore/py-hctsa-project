import numpy as np
from numpy import histogram_bin_edges
from scipy.stats import gaussian_kde, kurtosis, skew, expon
from statsmodels.tsa.stattools import pacf
from scipy.optimize import curve_fit
from scipy.stats import mode as smode
from loguru import logger
from utilities import point_of_crossing, signChange, binpicker, histc, ZScore
from typing import Union
from numpy.typing import ArrayLike


def StickAngles(y : ArrayLike) -> dict:
    """
    Analysis of the line-of-sight angles between time series data pts. 

    Line-of-sight angles between time-series pts treat each time-series value as a stick 
    protruding from an opaque baseline level. Statistics are returned on the raw time series, 
    where sticks protrude from the zero-level, and the z-scored time series, where sticks
    protrude from the mean level of the time series.

    Note: Any features derived from the KDE will be different to the output from MATLAB 
    as a result of differences in how the KDE works in scipy vs MATLAB.
    This will affect the features: `pnsumabsdiff`, `symks_p`, `symks_n`, `ratmean_p`, `ratmean_n`.

    Parameters:
    -----------
    y : array-like
        The input time series

    Returns:
    --------
    out : dict
        A dictionary containing various statistics on the obtained sequence of angles.
    """
    y = np.asarray(y)
    # Split the time series into positive and negative parts
    ix = [np.where(y >= 0)[0], np.where(y < 0)[0]]
    n = [len(ix[0]), len(ix[1])]

    # Compute the stick angles
    angles = [[], []]
    for j in range(2):
        if n[j] > 1:
            diff_y = np.diff(y[ix[j]])
            diff_x = np.diff(ix[j])
            angles[j] = np.arctan(diff_y /diff_x)
    allAngles = np.concatenate(angles)

    # Initialise output dictionary
    out = {}
    out['std_p'] = np.nanstd(angles[0], ddof=1) 
    out['mean_p'] = np.nanmean(angles[0]) 
    out['median_p'] = np.nanmedian(angles[0])

    out['std_n'] = np.nanstd(angles[1], ddof=1)
    out['mean_n'] = np.nanmean(angles[1])
    out['median_n'] = np.nanmedian(angles[1])

    out['std'] = np.nanstd(allAngles, ddof=1)
    out['mean'] = np.nanmean(allAngles)
    out['median'] = np.nanmedian(allAngles)

    # difference between positive and negative angles
    # return difference in densities
    ksx = np.linspace(np.min(allAngles), np.max(allAngles), 200)
    if len(angles[0]) > 0 and len(angles[1]) > 0:
        kde = gaussian_kde(angles[0], bw_method='scott')
        ksy1 = kde(ksx)
        kde2 = gaussian_kde(angles[1], bw_method='scott')
        ksy2 = kde2(ksx)
        out['pnsumabsdiff'] = np.sum(np.abs(ksy1-ksy2))
    else:
        out['pnsumabsdiff'] = np.nan
    
    # how symmetric is the distribution of angles?
    if len(angles[0]) > 0:
        maxdev = np.max(np.abs(angles[0]))
        kde = gaussian_kde(angles[0], bw_method='scott')
        ksy1 = kde(np.linspace(-maxdev, maxdev, 201))
        #print(ksy1[101:])
        out['symks_p'] = np.sum(np.abs(ksy1[:100] - ksy1[101:][::-1]))
        out['ratmean_p'] = np.mean(angles[0][angles[0] > 0])/np.mean(angles[0][angles[0] < 0])
    else:
        out['symks_p'] = np.nan
        out['ratmean_p'] = np.nan
    
    if len(angles[1]) > 0:
        maxdev = np.max(np.abs(angles[1]))
        kde = gaussian_kde(angles[1], bw_method='scott')
        ksy2 = kde(np.linspace(-maxdev, maxdev, 201))
        out['symks_n'] = np.sum(np.abs(ksy2[:100] - ksy2[101:][::-1]))
        out['ratmean_n'] = np.mean(angles[1][angles[1] > 0])/np.mean(angles[1][angles[1] < 0])
    else:
        out['symks_n'] = np.nan
        out['ratmean_n'] = np.nan
    
    # z-score
    zangles = []
    zangles.append(ZScore(angles[0]))
    zangles.append(ZScore(angles[1]))
    zallAngles = ZScore(allAngles)

    # how stationary are the angle sets?

    # there are positive angles
    if len(zangles[0]) > 0:
        # StatAv2
        out['statav2_p_m'], out['statav2_p_s'] = _SUB_statav(zangles[0], 2)
        # StatAv3
        out['statav3_p_m'], out['statav3_p_s'] = _SUB_statav(zangles[0], 3)
        # StatAv4
        out['statav4_p_m'], out['statav4_p_s'] = _SUB_statav(zangles[0], 4)
        # StatAv5
        out['statav5_p_m'], out['statav5_p_s'] = _SUB_statav(zangles[0], 5)
    else:
        out['statav2_p_m'], out['statav2_p_s'] = np.nan, np.nan
        out['statav3_p_m'], out['statav3_p_s'] = np.nan, np.nan
        out['statav4_p_m'], out['statav4_p_s'] = np.nan, np.nan
        out['statav5_p_m'], out['statav5_p_s'] = np.nan, np.nan
    
    # there are negative angles
    if len(zangles[1]) > 0:
        # StatAv2
        out['statav2_n_m'], out['statav2_n_s'] = _SUB_statav(zangles[1], 2)
        # StatAv3
        out['statav3_n_m'], out['statav3_n_s'] = _SUB_statav(zangles[1], 3)
        # StatAv4
        out['statav4_n_m'], out['statav4_n_s'] = _SUB_statav(zangles[1], 4)
        # StatAv5
        out['statav5_n_m'], out['statav5_n_s'] = _SUB_statav(zangles[1], 5)
    else:
        out['statav2_n_m'], out['statav2_n_s'] = np.nan, np.nan
        out['statav3_n_m'], out['statav3_n_s'] = np.nan, np.nan
        out['statav4_n_m'], out['statav4_n_s'] = np.nan, np.nan
        out['statav5_n_m'], out['statav5_n_s'] = np.nan, np.nan
    
    # All angles
    
    # StatAv2
    out['statav2_all_m'], out['statav2_all_s'] = _SUB_statav(zallAngles, 2)
    # StatAv3
    out['statav3_all_m'], out['statav3_all_s'] = _SUB_statav(zallAngles, 3)
    # StatAv4
    out['statav4_all_m'], out['statav4_all_s'] = _SUB_statav(zallAngles, 4)
    # StatAv5
    out['statav5_all_m'], out['statav5_all_s'] = _SUB_statav(zallAngles, 5)
    
    # correlations? 
    if len(zangles[0]) > 0:
        out['tau_p'] = FirstCrossing(zangles[0], 'ac', 0, 'continuous')
        out['ac1_p'] = AutoCorr(zangles[0], 1, 'Fourier')[0]
        out['ac2_p'] = AutoCorr(zangles[0], 2, 'Fourier')[0]
    else:
        out['tau_p'] = np.nan
        out['ac1_p'] = np.nan
        out['ac2_p'] = np.nan
    
    if len(zangles[1]) > 0:
        out['tau_n'] = FirstCrossing(zangles[1], 'ac', 0, 'continuous')
        out['ac1_n'] = AutoCorr(zangles[1], 1, 'Fourier')[0]
        out['ac2_n'] = AutoCorr(zangles[1], 2, 'Fourier')[0]
    else:
        out['tau_n'] = np.nan
        out['ac1_n'] = np.nan
        out['ac2_n'] = np.nan
    
    out['tau_all'] = FirstCrossing(zallAngles, 'ac', 0, 'continuous')
    out['ac1_all'] = AutoCorr(zallAngles, 1, 'Fourier')[0]
    out['ac2_all'] = AutoCorr(zallAngles, 2, 'Fourier')[0]


    # What does the distribution look like? 
    
    # Some quantiles and moments
    if len(zangles[0]) > 0:
        out['q1_p'] = np.quantile(zangles[0], 0.01, method='hazen')
        out['q10_p'] = np.quantile(zangles[0], 0.1, method='hazen')
        out['q90_p'] = np.quantile(zangles[0], 0.9, method='hazen')
        out['q99_p'] = np.quantile(zangles[0], 0.99, method='hazen')
        out['skewness_p'] = skew(angles[0])
        out['kurtosis_p'] = kurtosis(angles[0], fisher=False)
    else:
        out['q1_p'], out['q10_p'], out['q90_p'], out['q99_p'], \
            out['skewness_p'], out['kurtosis_p'] = np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan
    
    if len(zangles[1]) > 0:
        out['q1_n'] = np.quantile(zangles[1], 0.01, method='hazen')
        out['q10_n'] = np.quantile(zangles[1], 0.1, method='hazen')
        out['q90_n'] = np.quantile(zangles[1], 0.9, method='hazen')
        out['q99_n'] = np.quantile(zangles[1], 0.99, method='hazen')
        out['skewness_n'] = skew(angles[1])
        out['kurtosis_n'] = kurtosis(angles[1], fisher=False)
    else:
        out['q1_n'], out['q10_n'], out['q90_n'], out['q99_n'], \
            out['skewness_n'], out['kurtosis_n'] = np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan
    
    F_quantz = lambda x : np.quantile(zallAngles, x, method='hazen')
    out['q1_all'] = F_quantz(0.01)
    out['q10_all'] = F_quantz(0.1)
    out['q90_all'] = F_quantz(0.9)
    out['q99_all'] = F_quantz(0.99)
    out['skewness_all'] = skew(allAngles)
    out['kurtosis_all'] = kurtosis(allAngles, fisher=False)

    return out

def _SUB_statav(x, n):
    # helper function
    NN = len(x)
    if NN < 2 * n: # not long enough
        statavmean = np.nan
        statavstd = np.nan
    x_buff = _buffer(x, int(np.floor(NN/n)))
    if x_buff.shape[1] > n:
        # remove final pt
        x_buff = x_buff[:, :n]
    
    statavmean = np.std(np.mean(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)
    statavstd = np.std(np.std(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)

    return statavmean, statavstd

def _buffer(X, n, p=0, opt=None):
    # helper function
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

def FZCGLSCF(y: ArrayLike, alpha: Union[float, int], beta: Union[float, int], maxtau: Union[int, None] = None) -> float:
    """
    The first zero-crossing of the generalized self-correlation function.

    Returns the first zero-crossing of the generalized self-correlation function (GLSCF)
    introduced by Queirós and Moyano (2007). The function calculates the GLSCF at 
    increasing time delays until it finds a zero crossing, and returns this lag value.

    Uses GLSCF to calculate the generalized self-correlations at each lag.

    Parameters
    ----------
    y : array_like
        The input time series
    alpha : float 
        The parameter alpha for GLSCF calculation. Must be non-zero.
    beta : float
        The parameter beta for GLSCF calculation. Must be non-zero.
    maxtau : int, optional
        Maximum time delay to search up to. If None, uses the time-series length.
        Default is None.

    Returns
    -------
    float
        The time lag τ of the first zero-crossing of the GLSCF.

    References
    ----------
    .. [1] Queirós, S.M.D., Moyano, L.G. (2007) "Yet on statistical properties of 
           traded volume: Correlation and mutual information at different value magnitudes"
           Physica A, 383(1), pp. 10-15.
           DOI: 10.1016/j.physa.2007.04.068
    """
    y = np.asarray(y)
    N = len(y)

    if maxtau is None:
        maxtau = N
    
    glscfs = np.zeros(maxtau)

    for i in range(1, maxtau+1):
        tau = i

        glscfs[i-1] = GLSCF(y, alpha, beta, tau)
        if (i > 1) and (glscfs[i-1]*glscfs[i-2] < 0):
            # Draw a straight line between these two and look at where it hits zero
            out = i - 1 + glscfs[i-1]/(glscfs[i-1]-glscfs[i-2])
            return out
    
    return maxtau

def GLSCF(y : ArrayLike, alpha : float, beta : float, tau : Union[int, str] = 'tau') -> float:
    """
    Compute the generalized linear self-correlation function (GLSCF) of a time series.

    This function implements the GLSCF as introduced by Queirós and Moyano (2007) to analyze
    correlations in the magnitude of time series values at different scales. The GLSCF 
    generalizes traditional autocorrelation by applying different exponents to earlier and 
    later time points.

    The function is defined as:
        GLSCF = (E[|x(t)|^α |x(t+τ)|^β] - E[|x(t)|^α]E[|x(t+τ)|^β]) / 
                (σ(|x(t)|^α)σ(|x(t+τ)|^β))
    where E[] denotes expectation and σ() denotes standard deviation.

    Parameters
    ----------
    y : array_like
        The input time series
    alpha : float 
        Exponent applied to the earlier time point x(t). Must be non-zero.
    beta : float
        Exponent applied to the later time point x(t+τ). Must be non-zero.
    tau : Union[int, str], optional
        The time delay (lag) between points. If 'tau', uses first zero-crossing
        of autocorrelation function. Default is 'tau'.

    Returns
    -------
    float
        The GLSCF value at the specified lag τ

    References
    ----------
    .. [1] Queirós, S.M.D., Moyano, L.G. (2007) "Yet on statistical properties of 
           traded volume: Correlation and mutual information at different value magnitudes"
           Physica A, 383(1), pp. 10-15.
           DOI: 10.1016/j.physa.2007.04.068
    """
    # Set tau to first zero-crossing of the autocorrelation function with the input 'tau'
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Take magnitudes of time-delayed versions of the time series
    y1 = np.abs(y[:-tau])
    y2 = np.abs(y[tau:])


    p1 = np.mean(np.multiply((y1 ** alpha), (y2 ** beta)))
    p2 = np.multiply(np.mean(y1 ** alpha), np.mean(y2 ** beta))
    p3 = np.sqrt(np.mean(y1 ** (2*alpha)) - (np.mean(y1 ** alpha))**2)
    p4 = np.sqrt(np.mean(y2 ** (2*beta)) - (np.mean(y2 ** beta))**2)

    glscf = (p1 - p2) / (p3 * p4)

    return glscf

# def Embed2Shapes(y : ArrayLike, tau : Union[str, int, None] = 'tau', shape : str = 'circle', r : float = 1.0) -> dict:
#     """
#     Shape-based statistics in a 2-d embedding space.

#     Takes a shape and places it on each point in the two-dimensional time-delay
#     embedding space sequentially. This function counts the points inside this shape
#     as a function of time, and returns statistics on this extracted time series.

#     Parameters:
#     -----------
#     y : array_like
#         The input time-series as a (z-scored) column vector.
#     tau : int or str, optional
#         The time-delay. If 'tau', it's set to the first zero crossing of the autocorrelation function.
#     shape : str, optional
#         The shape to use. Currently only 'circle' is supported.
#     r : float, optional
#         The radius of the circle.

#     Returns:
#     --------
#     dict
#         A dictionary containing various statistics of the constructed time series.
#     """
#     y = np.asarray(y)
#     if tau == 'tau':
#         tau = FirstCrossing(y, 'ac', 0, 'discrete')
#         # cannot set time delay > 10% of the length of the time series...
#         if tau > len(y)/10:
#             tau = int(np.floor(len(y)/10))
        
#     # Create the recurrence space, populated by points m
#     m = np.column_stack((y[:-tau], y[tau:]))
#     N = len(m)

#     # Start the analysis
#     counts = np.zeros(N)
#     if shape == 'circle':
#         # Puts a circle around each point in the embedding space in turn
#         # counts how many pts are inside this shape, looks at the time series thus formed
#         for i in range(N): # across all pts in the time series
#             m_c = m - m[i] # pts wrt current pt i
#             m_c_d = np.sum(m_c**2, axis=1) # Euclidean distances from pt i
#             counts[i] = np.sum(m_c_d <= r**2) # number of pts enclosed in a circle of radius r
#     else:
#         raise ValueError(f"Unknown shape '{shape}'")
    
#     counts -= 1 # ignore self counts

#     if np.all(counts == 0):
#         print("No counts detected!")
#         return np.nan

#     # Return basic statistics on the counts
#     out = {}
#     out['ac1'] = AutoCorr(counts, 1, 'Fourier')[0]
#     out['ac2'] = AutoCorr(counts, 2, 'Fourier')[0]
#     out['ac3'] = AutoCorr(counts, 3, 'Fourier')[0]
#     out['tau'] = FirstCrossing(counts, 'ac', 0, 'continuous')
#     out['max'] = np.max(counts)
#     out['std'] = np.std(counts, ddof=1)
#     out['median'] = np.median(counts)
#     out['mean'] = np.mean(counts)
#     out['iqr'] = np.percentile(counts, 75, method='hazen') - np.percentile(counts, 25, method='hazen')
#     out['iqronrange'] = out['iqr']/np.ptp(counts)

#     # distribution - using sqrt binning method
#     numBinsToUse = int(np.ceil(np.sqrt(len(counts)))) # supposed to be what MATLAB uses for 'sqrt' option.
#     minX, maxX = np.min(counts), np.max(counts)
#     binWidthEst = (maxX - minX)/numBinsToUse
#     binEdges = binpicker(minX, maxX, nbins=None, bindwidthEst=binWidthEst)
#     print(binEdges)
#     binCounts = histc(counts, binEdges)
#     # normalise bin counts
#     binCountsNorm = np.divide(binCounts, np.sum(binCounts))
#     # get bin centres
#     binCentres = (binEdges[:-1] + binEdges[1:]) / 2
#     out['mode_val'] = np.max(binCountsNorm)
#     out['mode'] = binCentres[np.argmax(binCountsNorm)]
#     # histogram entropy
#     out['hist_ent'] = np.sum(binCountsNorm[binCountsNorm > 0] * np.log(binCountsNorm[binCountsNorm > 0]))

#     # Stationarity measure for fifths of the time series
#     afifth = int(np.floor(N/5))
#     buffer_m = np.array([counts[i*afifth:(i+1)*afifth] for i in range(5)])
#     out['statav5_m'] = np.std(np.mean(buffer_m, axis=1), ddof=1) / np.std(counts, ddof=1)
#     out['statav5_s'] = np.std(np.std(buffer_m, axis=1, ddof=1), ddof=1) / np.std(counts, ddof=1)

#     return out

# def Embed2Dist(y : ArrayLike, tau : Union[None, str] = None) -> dict:
#     """
#     Analyzes distances in a 2-dim embedding space of a time series.

#     Returns statistics on the sequence of successive Euclidean distances between
#     points in a two-dimensional time-delay embedding space with a given
#     time-delay, tau.

#     Outputs include the autocorrelation of distances, the mean distance, the
#     spread of distances, and statistics from an exponential fit to the
#     distribution of distances.

#     Parameters:
#     y (array-like): A z-scored column vector representing the input time series.
#     tau (int, optional): The time delay. If None, it's set to the first minimum of the autocorrelation function.

#     Returns:
#     dict: A dictionary containing various statistics of the embedding.
#     """
#     y = np.asarray(y)
#     N = len(y) # time-series length

#     if tau is None:
#         tau = 'tau' # set to the first minimum of autocorrelation function
    
#     if tau == 'tau':
#         tau = FirstCrossing(y, 'ac', 0, 'discrete')
#         if tau > N / 10:
#             tau = N//10

#     # Make sure the time series is a column vector
#     y = np.asarray(y).reshape(-1, 1)

#     # Construct a 2-dimensional time-delay embedding (delay of tau)
#     m = np.hstack((y[:-tau], y[tau:]))

#     # Calculate Euclidean distances between successive points in this space, d:
#     out = {}
#     d = np.sqrt(np.sum(np.diff(m, axis=0)**2, axis=1))
    
#     # Calculate autocorrelations
#     out['d_ac1'] = AutoCorr(d, 1, 'Fourier')[0] # lag 1 ac
#     out['d_ac2'] = AutoCorr(d, 2, 'Fourier')[0] # lag 2 ac
#     out['d_ac3'] = AutoCorr(d, 3, 'Fourier')[0] # lag 3 ac

#     out['d_mean'] = np.mean(d) # Mean distance
#     out['d_median'] = np.median(d) # Median distance
#     out['d_std'] = np.std(d, ddof=1) # Standard deviation of distances
#     # need to use Hazen method of computing percentiles to get IQR consistent with MATLAB
#     q75 = np.percentile(d, 75, method='hazen')
#     q25 = np.percentile(d, 25, method='hazen')
#     iqr_val = q75 - q25
#     out['d_iqr'] = iqr_val # Interquartile range of distances
#     out['d_max'] = np.max(d) # Maximum distance
#     out['d_min'] = np.min(d) # Minimum distance
#     out['d_cv'] = np.mean(d) / np.std(d, ddof=1) # Coefficient of variation of distances

#     # Empirical distances distribution often fits Exponential distribution quite well
#     # Fit to all values (often some extreme outliers, but oh well)
#     l = 1 / np.mean(d)
#     nlogL = -np.sum(expon.logpdf(d, scale=1/l))
#     out['d_expfit_nlogL'] = nlogL

#     # Calculate histogram
#     # unable to get exact equivalence with MATLAB's histcount function, although numpy's histogram_edges gets very close...
#     #print(len(d))
#     #bin_edges = binpicker(d.min(), d.max(), nbins=27)
#     #print(bin_edges)
#     #N = histc(d, bin_edges)
#     N, bin_edges = _histcounts(d, bins='auto', normalization='probability')
#     #print(bin_edges)
#     bin_centers = np.mean(np.vstack([bin_edges[:-1], bin_edges[1:]]), axis=0)
#     exp_fit = expon.pdf(bin_centers, scale=1/l)
#     out['d_expfit_meandiff'] = np.mean(np.abs(N - exp_fit))

#     return out

# def _histcounts(x, bins=None, binEdges=None, normalization='probability'):
#     x = np.asarray(x).flatten()
#     if binEdges is not None:
#         edges = np.asarray(binEdges)
#     elif bins is None or bins == 'auto':
#         edges = histogram_bin_edges(x, bins='auto')
#     elif isinstance(bins, int):
#         edges = np.linspace(np.min(x), np.max(x), bins + 1)
#     else:
#         raise ValueError("Invalid bins parameter")

#     n, _ = np.histogram(x, bins=edges)
    
#     # Apply normalization
#     if normalization != 'count':
#         bin_widths = np.diff(edges)
#         if normalization == 'countdensity':
#             n = n / bin_widths
#         elif normalization == 'cumcount':
#             n = np.cumsum(n)
#         elif normalization == 'probability':
#             n = n / len(x)
#         elif normalization == 'percentage':
#             n = (100 * n) / len(x)
#         elif normalization == 'pdf':
#             n = n / (len(x) * bin_widths)
#         elif normalization == 'cdf':
#             n = np.cumsum(n / len(x))
#         else:
#             raise ValueError(f"Invalid normalization method: {normalization}")
    
#     return n, edges

def Embed2Basic(y : ArrayLike, tau : Union[int, str] = 1) -> dict:
    """
    Point density statistics in a 2-d embedding space.

    Computes a set of point-density statistics in a plot of y_i against y_{i-tau}. The function 
    calculates the density of points near various geometric shapes in the embedding space, 
    including diagonals, parabolas, rings, and circles.

    Parameters
    -----------
    y : array_like
        The input time series.
    tau : int or str, optional
        The time lag (can be set to 'tau' to set the time lag to the first zero
        crossing of the autocorrelation function). Default is 1.

    Returns
    --------
    out : dict
        Dictionary containing various point density statistics.
    """
    y = np.asarray(y)
    if tau == 'tau':
        # Make tau the first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')

    xt = y[:-tau]  # part of the time series
    xtp = y[tau:]  # time-lagged time series
    N = len(y) - tau  # Length of each time series subsegment

    out = {}

    # Points in a thick bottom-left -- top-right diagonal
    out['updiag01'] = np.divide(np.sum(np.abs(xtp - xt) < 0.1), N)
    out['updiag05'] = np.divide(np.sum(np.abs(xtp - xt) < 0.5), N)

    # Points in a thick bottom-right -- top-left diagonal
    out['downdiag01'] = np.divide(np.sum(np.abs(xtp + xt) < 0.1), N)
    out['downdiag05'] = np.divide(np.sum(np.abs(xtp + xt) < 0.5), N)

    # Ratio of these
    out['ratdiag01'] = np.divide(out['updiag01'], out['downdiag01'])
    out['ratdiag05'] = np.divide(out['updiag05'], out['downdiag05'])

    # In a thick parabola concave up
    out['parabup01'] = np.divide(np.sum(np.abs(xtp - xt**2) < 0.1), N)
    out['parabup05'] = np.divide(np.sum(np.abs(xtp - xt**2) < 0.5), N)

    # In a thick parabola concave down
    out['parabdown01'] = np.divide(np.sum(np.abs(xtp + xt**2) < 0.1), N)
    out['parabdown05'] = np.divide(np.sum(np.abs(xtp + xt**2) < 0.5), N)

    # In a thick parabola concave up, shifted up 1
    out['parabup01_1'] = np.divide(np.sum(np.abs(xtp - (xt**2 + 1)) < 0.1), N)
    out['parabup05_1'] = np.divide(np.sum(np.abs(xtp - (xt**2 + 1)) < 0.5), N)

    # In a thick parabola concave down, shifted up 1 
    out['parabdown01_1'] = np.divide(np.sum(np.abs(xtp + (xt**2 - 1)) < 0.1), N)
    out['parabdown05_1'] = np.divide(np.sum(np.abs(xtp + (xt**2 - 1)) < 0.5), N)

    # In a thick parabola concave up, shifted down 1
    out['parabup01_n1'] = np.divide(np.sum(np.abs(xtp - (xt**2 - 1)) < 0.1), N)
    out['parabup05_n1'] = np.divide(np.sum(np.abs(xtp - (xt**2 - 1)) < 0.5), N)

    # In a thick parabola concave down, shifted down 1
    out['parabdown01_n1'] = np.divide(np.sum(np.abs(xtp + (xt**2 + 1)) < 0.1), N)
    out['parabdown05_n1'] = np.divide(np.sum(np.abs(xtp + (xt**2 + 1)) < 0.5), N)

    # RINGS (points within a radius range)
    out['ring1_01'] = np.divide(np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.1), N)
    out['ring1_02'] = np.divide(np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.2), N)
    out['ring1_05'] = np.divide(np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.5), N)

    # CIRCLES (points inside a given circular boundary)
    out['incircle_01'] = np.divide(np.sum(xtp**2 + xt**2 < 0.1), N)
    out['incircle_02'] = np.divide(np.sum(xtp**2 + xt**2 < 0.2), N)
    out['incircle_05'] = np.divide(np.sum(xtp**2 + xt**2 < 0.5), N)
    out['incircle_1'] = np.divide(np.sum(xtp**2 + xt**2 < 1), N)
    out['incircle_2'] = np.divide(np.sum(xtp**2 + xt**2 < 2), N)
    out['incircle_3'] = np.divide(np.sum(xtp**2 + xt**2 < 3), N)
    
    incircle_values = [out['incircle_01'], out['incircle_02'], out['incircle_05'],
                       out['incircle_1'], out['incircle_2'], out['incircle_3']]
    out['medianincircle'] = np.median(incircle_values)
    out['stdincircle'] = np.std(incircle_values, ddof=1)
    
    return out

# def Embed2DAngleTau(y : ArrayLike, maxTau : int = 5) -> dict:
#     """
#     Angle autocorrelation in a 2-dimensional embedding space.

#     Investigates how the autocorrelation of angles between successive points in
#     the two-dimensional time-series embedding change as tau varies from
#     tau = 1, 2, ..., maxTau.

#     Parameters:
#     -----------
#     y (numpy.ndarray): Input time series (1D array)
#     maxTau (int): The maximum time lag to consider

#     Returns:
#     --------
#     dict: A dictionary containing various statistics
#     """
#     tauRange = np.arange(1, maxTau + 1)
#     numTau = len(tauRange)

#     # Ensure y is a column vector
#     y = np.atleast_2d(y)
#     if y.shape[0] < y.shape[1]:
#         y = y.T

#     stats_store = np.zeros((3, numTau))

#     for i, tau in enumerate(tauRange):
#         m = np.column_stack((y[:-tau], y[tau:]))
#         theta = np.diff(m[:, 1]) / np.diff(m[:, 0])
#         theta = np.arctan(theta)  # measured as deviation from the horizontal

#         if len(theta) == 0:
#             raise ValueError(f'Time series (N={len(y)}) too short for embedding')

#         stats_store[0, i] = AutoCorr(theta, 1, 'Fourier')[0]
#         stats_store[1, i] = AutoCorr(theta, 2, 'Fourier')[0]
#         stats_store[2, i] = AutoCorr(theta, 3, 'Fourier')[0]
#         #print(stats_store)
    
#     # Compute output statistics
#     out = {
#         'ac1_thetaac1': AutoCorr(stats_store[0, :], 1, 'Fourier')[0],
#         'ac1_thetaac2': AutoCorr(stats_store[1, :], 1, 'Fourier')[0],
#         'ac1_thetaac3': AutoCorr(stats_store[2, :], 1, 'Fourier')[0],
#         'mean_thetaac1': np.mean(stats_store[0, :]),
#         'max_thetaac1': np.max(stats_store[0, :]),
#         'min_thetaac1': np.min(stats_store[0, :]),
#         'mean_thetaac2': np.mean(stats_store[1, :]),
#         'max_thetaac2': np.max(stats_store[1, :]),
#         'min_thetaac2': np.min(stats_store[1, :]),
#         'mean_thetaac3': np.mean(stats_store[2, :]),
#         'max_thetaac3': np.max(stats_store[2, :]),
#         'min_thetaac3': np.min(stats_store[2, :]),
#     }

#     out['meanrat_thetaac12'] = out['mean_thetaac1'] / out['mean_thetaac2']
#     out['diff_thetaac12'] = np.sum(np.abs(stats_store[1, :] - stats_store[0, :]))

#     return out

# def CompareMinAMI(y : ArrayLike, binMethod : str = 'std1', numBins : int = 10) -> dict:
#     """
#     Variability in first minimum of automutual information.

#     Finds the first minimum of the automutual information by various different
#     estimation methods, and sees how this varies over different coarse-grainings
#     of the time series.

#     Args:
#     y (array-like): The input time series
#     binMethod (str): The method for estimating mutual information (input to CO_HistogramAMI)
#     numBins (int or array-like): The number of bins for the AMI estimation to compare over

#     Returns:
#     dict: A dictionary containing various statistics on the set of first minimums 
#           of the automutual information function
#     """
#     y = np.asarray(y)
#     N = len(y)
#     # Range of time lags to consider
#     tauRange = np.arange(0, int(np.ceil(N/2))+1)
#     numTaus = len(tauRange)

#     # range of bin numbers to consider
#     if isinstance(numBins, int):
#         numBins = [numBins]
    
#     numBinsRange = len(numBins)
#     amiMins = np.zeros(numBinsRange)

#     # Calculate automutual information
#     for i in range(numBinsRange):  # vary over number of bins in histogram
#         amis = np.zeros(numTaus)
#         for j in range(numTaus):  # vary over time lags, tau
#             amis[j] = HistogramAMI(y, tauRange[j], binMethod, numBins[i])
#             if (j > 1) and ((amis[j] - amis[j-1]) * (amis[j-1] - amis[j-2]) < 0):
#                 amiMins[i] = tauRange[j-1]
#                 break
#         if amiMins[i] == 0:
#             amiMins[i] = tauRange[-1]
#     # basic statistics
#     out = {}
#     out['min'] = np.min(amiMins)
#     out['max'] = np.max(amiMins)
#     out['range'] = np.ptp(amiMins)
#     out['median'] = np.median(amiMins)
#     out['mean'] = np.mean(amiMins)
#     out['std'] = np.std(amiMins, ddof=1) # will return NaN for single values instead of 0
#     out['nunique'] = len(np.unique(amiMins))
#     out['mode'], out['modef'] = smode(amiMins)
#     out['modef'] = out['modef']/numBinsRange

#     # converged value? 
#     out['conv4'] = np.mean(amiMins[-5:])

#     # look for peaks (local maxima)
#     # % local maxima above 1*std from mean
#     # inspired by curious result of periodic maxima for periodic signal with
#     # bin size... ('quantiles', [2:80])
#     diff_ami_mins = np.diff(amiMins[:-1])
#     positive_diff_indices = np.where(diff_ami_mins > 0)[0]
#     sign_change_indices = signChange(diff_ami_mins, 1)

#     # Find the intersection of positive_diff_indices and sign_change_indices
#     loc_extr = np.intersect1d(positive_diff_indices, sign_change_indices) + 1
#     above_threshold_indices = np.where(amiMins > out['mean'] + out['std'])[0]
#     big_loc_extr = np.intersect1d(above_threshold_indices, loc_extr)

#     # Count the number of elements in big_loc_extr
#     out['nlocmax'] = len(big_loc_extr)

#     return out

def PartialAutoCorr(y : ArrayLike, maxTau : int = 10, whatMethod : str = 'ols') -> dict:
    """
    Compute the partial autocorrelation of an input time series.
    
    This function calculates the partial autocorrelation function (PACF) up to a specified 
    lag using either ordinary least squares or Yule-Walker equations. This is part of 
    the Correlation operations from hctsa.

    Parameters
    ----------
    y : array-like
        The input time series as a scalar column vector
    maxTau : int, optional
        The maximum time-delay to compute PACF values for (default=10)
    whatMethod : {'ols', 'Yule-Walker'}, optional
        Method to compute partial autocorrelation (default='ols'):
        - 'ols': Ordinary least squares regression
        - 'Yule-Walker': Yule-Walker equations method

    Returns
    -------
    dict
        Dictionary containing partial autocorrelations for each lag, with keys:
        - 'pac_1': PACF at lag 1
        - 'pac_2': PACF at lag 2
        ...up to maxTau
    """
    y = np.asarray(y)
    N = len(y)
    if maxTau <= 0:
        raise ValueError('Negative or zero time lags not applicable')

    method_map = {'ols': 'ols', 'Yule-Walker': 'ywm'} 
    if whatMethod not in method_map:
        raise ValueError(f"Invalid method: {whatMethod}. Use 'ols' or 'Yule-Walker'.")

    # Compute partial autocorrelation
    pacf_values = pacf(y, nlags=maxTau, method=method_map[whatMethod])

    # Create output dictionary
    out = {}
    for i in range(1, maxTau + 1):
        out[f'pac_{i}'] = pacf_values[i]

    return out

def HistogramAMI(y : ArrayLike, tau : Union[str, int, ArrayLike] = 1, meth : str = 'even', numBins : int = 10) -> dict:
    """
    The automutual information of the distribution using histograms.

    Computes the automutual information between a time series and its time-delayed version
    using different methods for binning the data.

    Parameters
    ----------
    y : array-like
        The input time series
    tau : int, list, or str, optional
        The time-lag(s) (default: 1)
        Can be an integer time lag, list of time lags, or 'ac'/'tau' to use
        first zero-crossing of autocorrelation function
    meth : str, optional
        The method for binning data (default: 'even'):
        - 'even': evenly-spaced bins through the range
        - 'std1': bins extending to ±1 standard deviation from mean
        - 'std2': bins extending to ±2 standard deviations from mean
        - 'quantiles': equiprobable bins using quantiles
    numBins : int, optional
        The number of bins to use (default: 10)

    Returns
    -------
    Union[float, dict]
        If single tau: The automutual information value
        If multiple taus: Dictionary of automutual information values
    """
    # Use first zero crossing of the ACF as the time lag
    y = np.asarray(y)
    if isinstance(tau, str) and tau in ['ac', 'tau']:
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Bins for the data
    # same for both -- assume same distribution (true for stationary processes, or small lags)
    if meth == 'even':
        b = np.linspace(np.min(y), np.max(y), numBins + 1)
        # Add increment buffer to ensure all points are included
        inc = 0.1
        b[0] -= inc
        b[-1] += inc
    elif meth == 'std1': # bins out to +/- 1 std
        b = np.linspace(-1, 1, numBins + 1)
        if np.min(y) < -1:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 1:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'std2': # bins out to +/- 2 std
        b = np.linspace(-2, 2, numBins + 1)
        if np.min(y) < -2:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 2:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'quantiles': # use quantiles with ~equal number in each bin
        b = np.quantile(y, np.linspace(0, 1, numBins + 1), method='hazen')
        b[0] -= 0.1
        b[-1] += 0.1
    else:
        raise ValueError(f"Unknown method '{meth}'")
    
    # Sometimes bins can be added (e.g., with std1 and std2), so need to redefine numBins
    numBins = len(b) - 1

    # Form the time-delay vectors y1 and y2
    if not isinstance(tau, (list, np.ndarray)):
        # if only single time delay as integer, make into a one element list
        tau = [tau]

    amis = np.zeros(len(tau))

    #print(amis)
    #print(b)
    #print(tau)

    for i, t in enumerate(tau):
        if t == 0:
            # for tau = 0, y1 and y2 are identical to y
            y1 = y2 = y
        else:
            y1 = y[:-t]
            y2 = y[t:]
        # Joint distribution of y1 and y2
        pij, _, _ = np.histogram2d(y1, y2, bins=(b, b))
        pij = pij[:numBins, :numBins]  # joint
        pij = pij / np.sum(pij)  # normalize
        pi = np.sum(pij, axis=1)  # marginal
        pj = np.sum(pij, axis=0)  # other marginal

        pii = np.tile(pi, (numBins, 1)).T
        pjj = np.tile(pj, (numBins, 1))

        r = pij > 0  # Defining the range in this way, we set log(0) = 0
        amis[i] = np.sum(pij[r] * np.log(pij[r] / pii[r] / pjj[r]))

    if len(tau) == 1:
        return amis[0]
    else:
        return {f'ami{i+1}': ami for i, ami in enumerate(amis)}

# def _nk_hist2d(x, y, xedges, yedges):
#     """Extract the number of joint events - (x,y) data value pairs that fall 
#     in each bin of the grid defined by xedges and yedges."""
#     if len(x) != len(y):
#         raise ValueError("The length of x and y should be the same.")
#     xn, xbin = histc(x, xedges)
#     yn, ybin = histc(y, yedges)

# def NonlinearAutoCorr(y : ArrayLike, taus : ArrayLike, doAbs : Union[bool, None] = None):
#     """
#     A custom nonlinear autocorrelation of a time series.

#     Nonlinear autocorrelations are of the form:
#     <x_i x_{i-tau_1} x{i-tau_2}...>
#     The usual two-point autocorrelations are
#     <x_i.x_{i-tau}>

#     Parameters:
#     y (array-like): Should be the z-scored time series (Nx1 vector)
#     taus (array-like): Should be a vector of the time delays (mx1 vector)
#         e.g., [2] computes <x_i x_{i-2}>
#         e.g., [1,2] computes <x_i x_{i-1} x_{i-2}>
#         e.g., [1,1,3] computes <x_i x_{i-1}^2 x_{i-3}>
#         e.g., [0,0,1] computes <x_i^3 x_{i-1}>
#     do_abs (bool, optional): If True, takes an absolute value before taking the final mean.
#         Useful for an odd number of contributions to the sum.
#         Default is to do this for odd numbers anyway, if not specified.

#     Returns:
#     out (float): The computed nonlinear autocorrelation.

#     Notes:
#     - For odd numbers of regressions (i.e., even number length taus vectors)
#       the result will be near zero due to fluctuations below the mean;
#       even for highly-correlated signals. (do_abs)
#     - do_abs = True is really a different operation that can't be compared with
#       the values obtained from taking do_abs = False (i.e., for odd lengths of taus)
#     - It can be helpful to look at nonlinearAC at each iteration.
#     """
#     y = np.asarray(y)
#     if doAbs == None:
#         if len(taus) % 2 == 1:
#             doAbs = 0
#         else:
#             doAbs = 1

#     N = len(y)
#     tmax = np.max(taus)

#     nlac = y[tmax:N]

#     for i in taus:
#         nlac = np.multiply(nlac,y[tmax - i:N - i])

#     if doAbs:
#         out = np.mean(np.absolute(nlac))

#     else:
#         out = np.mean(nlac)

#     return out

def AutoCorrShape(y : ArrayLike, stopWhen : Union[int, str] = 'posDrown') -> dict:
    """
    How the autocorrelation function changes with the time lag.

    Outputs include the number of peaks, and autocorrelation in the
    autocorrelation function (ACF) itself.

    Parameters
    -----------
    y : array_like
        The input time series
    stopWhen : str or int, optional
        The criterion for the maximum lag to measure the ACF up to.
        Default is 'posDrown'.

    Returns
    --------
    dict
        A dictionary containing various metrics about the autocorrelation function.
    """
    y = np.asarray(y)
    N = len(y)

    # Only look up to when two consecutive values are under the significance threshold
    th = 2 / np.sqrt(N)  # significance threshold

    # Calculate the autocorrelation function, up to a maximum lag, length of time series (hopefully it's cropped by then)
    acf = []

    # At what lag does the acf drop to zero, Ndrown (by my definition)?
    if isinstance(stopWhen, int):
        taus = list(range(0, stopWhen+1))
        acf = AutoCorr(y, taus, 'Fourier')
        Ndrown = stopWhen
        
    elif stopWhen in ['posDrown', 'drown', 'doubleDrown']:
        # Compute ACF up to a given threshold:
        Ndrown = 0 # the point at which ACF ~ 0
        if stopWhen == 'posDrown':
            # stop when ACF drops below threshold, th
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0]
                if np.isnan(acf_val):
                    logger.warning("Weird time series (constant?)")
                    out = np.nan
                if acf_val < th:
                    # Ensure ACF is all positive
                    if acf_val > 0:
                        Ndrown = i
                        acf.append(acf_val)
                    else:
                        # stop at the previous point if not positive
                        Ndrown = i-1
                    # ACF has dropped below threshold, break the for loop...
                    break
                # hasn't dropped below thresh, append to list 
                acf.append(acf_val)
            # This should yield the initial, positive portion of the ACF.
            assert all(np.array(acf) > 0)
        elif stopWhen == 'drown':
            # Stop when ACF is very close to 0 (within threshold, th = 2/sqrt(N))
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0] # acf vector indicies are not lags
                # if positive and less than thresh
                if i > 0 and abs(acf_val) < th:
                    Ndrown = i
                    acf.append(acf_val)
                    break
                acf.append(acf_val)
        elif stopWhen == 'doubleDrown':
            # Stop at 2*tau, where tau is the lag where ACF ~ 0 (within 1/sqrt(N) threshold)
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0]
                if Ndrown > 0 and i == Ndrown * 2:
                    acf.append(acf_val)
                    break
                elif i > 1 and abs(acf_val) < th:
                    Ndrown = i
                acf.append(acf_val)
    else:
        raise ValueError(f"Unknown ACF decay criterion: '{stopWhen}'")

    acf = np.array(acf)
    Nac = len(acf)

    # Check for good behavior
    if np.any(np.isnan(acf)):
        # This is an anomalous time series (e.g., all constant, or conatining NaNs)
        out = np.nan
    
    out = {}
    out['Nac'] = Ndrown

    # Basic stats on the ACF
    out['sumacf'] = np.sum(acf)
    out['meanacf'] = np.mean(acf)
    if stopWhen != 'posDrown':
        out['meanabsacf'] = np.mean(np.abs(acf))
        out['sumabsacf'] = np.sum(np.abs(acf))

    # Autocorrelation of the ACF
    minPointsForACFofACF = 5 # can't take lots of complex stats with fewer than this

    if Nac > minPointsForACFofACF:
        out['ac1'] = AutoCorr(acf, 1, 'Fourier')[0]
        if all(acf > 0):
            out['actau'] = np.nan
        else:
            out['actau'] = AutoCorr(acf, FirstCrossing(acf, 'ac', 0, 'discrete'), 'Fourier')[0]

    else:
        out['ac1'] = np.nan
        out['actau'] = np.nan
    
    # Local extrema
    dacf = np.diff(acf)
    ddacf = np.diff(dacf)
    extrr = signChange(dacf, 1)
    sdsp = ddacf[extrr]

    # Proportion of local minima
    out['nminima'] = np.sum(sdsp > 0)
    out['meanminima'] = np.mean(sdsp[sdsp > 0])

    # Proportion of local maxima
    out['nmaxima'] = np.sum(sdsp < 0)
    out['meanmaxima'] = abs(np.mean(sdsp[sdsp < 0])) # must be negative: make it positive

    # Proportion of extrema
    out['nextrema'] = len(sdsp)
    out['pextrema'] = len(sdsp) / Nac

    # Fit exponential decay (only for 'posDrown', and if there are enough points)
    # Should probably only do this up to the first zero crossing...
    fitSuccess = False
    minPointsToFitExp = 4 # (need at least four points to fit exponential)

    if stopWhen == 'posDrown' and Nac >= minPointsToFitExp:
        # Fit exponential decay to (absolute) ACF:
        # (kind of only makes sense for the first positive period)
        expFunc = lambda x, b : np.exp(-b * x)
        try:
            popt, _ = curve_fit(expFunc, np.arange(Nac), acf, p0=0.5)
            fitSuccess = True
        except:
            fitSuccess = False
        
    if fitSuccess:
        bFit = popt[0] # fitted b
        out['decayTimescale'] = 1 / bFit
        expFit = expFunc(np.arange(Nac), bFit)
        residuals = acf - expFit
        out['fexpacf_r2'] = 1 - (np.sum(residuals**2) / np.sum((acf - np.mean(acf))**2))
        # had to fit a second exponential function with negative b to get same output as MATLAB for std residuals
        expFit2 = expFunc(np.arange(Nac), -bFit)
        residuals2 = acf - expFit2
        out['fexpacf_stdres'] = np.std(residuals2, ddof=1) 

    else:
        # Fit inappropriate (or failed): return nans for the relevant stats
        out['decayTimescale'] = np.nan
        out['fexpacf_r2'] = np.nan
        out['fexpacf_stdres'] = np.nan
    
    return out

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


def AutoCorr(y: ArrayLike, tau: Union[int, list] = 1, method: str = 'Fourier') -> Union[float, np.ndarray]:
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
        raise ValueError(f"Unknown autocorrelation estimation method {method}")
    
    return out

def FirstCrossing(y: ArrayLike, corr_fun: str = 'ac', threshold: float = 0.0, what_out: str = 'both') -> Union[dict, float]:
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
