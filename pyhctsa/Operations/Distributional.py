import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union
from scipy import stats
import warnings
from loguru import logger
from utilities import simple_binner, binpicker, histc
from Correlation import AutoCorr, FirstCrossing

def Withinp(x : ArrayLike, p : float = 1.0, meanOrMedian : str = 'mean') -> float:
    """
    Proportion of data points within p standard deviations of the mean or median.

    Parameters:
    -----------
    x (array-like): The input data vector
    p (float): The number (proportion) of standard deviations
    meanOrMedian (str): Whether to use units of 'mean' and standard deviation,
                          or 'median' and rescaled interquartile range

    Returns:
    --------
    float: The proportion of data points within p standard deviations

    Raises:
    ValueError: If mean_or_median is not 'mean' or 'median'
    """
    x = np.asarray(x)
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x, ddof=1)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        iqr_val = np.percentile(x, 75, method='hazen') - np.percentile(x, 25, method='hazen')
        sig = 1.35 * iqr_val
    else:
        raise ValueError(f"Unknown setting: '{meanOrMedian}'")

    # The withinp statistic:
    return np.divide(np.sum((x >= mu - p * sig) & (x <= mu + p * sig)), N)

def Unique(y : ArrayLike) -> float:
    """
    The proportion of the time series that are unique values.

    Parameters
    ----------
    y : array-like
        The input time series or data vector

    Returns
    -------
    float
        the proportion of time series that are unique values
    """
    x = np.asarray(y)
    return np.divide(len(np.unique(y)), len(y))

def TrimmedMean(y : ArrayLike, p_exclude : float = 0.0) -> float:
    """
    Mean of the trimmed time series.

    Returns the mean of the time series after removing a specified percentage of 
    the highest and lowest values. This is part of the Distributional operations 
    from hctsa, implementing DN_TrimmedMean.

    Parameters
    ----------
    y : array-like
        The input time series or data vector
    p_exclude : float, optional
        The percentage of highest and lowest values to exclude from the mean 
        calculation (default is 0.0, which gives the standard mean)

    Returns
    -------
    float
        The mean of the trimmed time series
    """
    y = np.asarray(y)
    p_exclude *= 0.01
    N = len(y)
    trim = int(np.round(N * p_exclude / 2))
    y = np.sort(y)

    out = np.mean(y[trim:N-trim])

    return out

def Spread(y : ArrayLike, spreadMeasure : str = 'std') -> float:
    """
    Measure of spread of the input time series.

    Returns the spread of the raw data vector using different statistical measures.
    This is part of the Distributional operations from hctsa, implementing DN_Spread.

    Parameters
    ----------
    y : array-like
        The input time series or data vector
    spreadMeasure : str, optional
        The spread measure to use (default is 'std'):
        - 'std': standard deviation
        - 'iqr': interquartile range 
        - 'mad': mean absolute deviation
        - 'mead': median absolute deviation

    Returns
    -------
    float
        The calculated spread measure
    """
    y = np.asarray(y)
    if spreadMeasure == 'std':
        out = np.std(y, ddof=1)
    elif spreadMeasure == 'iqr':
        # midpoint interpolation to match MATLAB implementation of IQR 
        out = stats.iqr(y, interpolation='midpoint')
    elif spreadMeasure == 'mad':
        # mean absolute deviation
        out = np.mean(np.absolute(y - np.mean(y, None)), None)
    elif spreadMeasure == 'mead':
        # median absolute deviation
        out = np.median(np.absolute(y - np.median(y, None)), None)
    else:
        raise ValueError('spreadMeasure must be one of std, iqr, mad or mead')
    return out

# def RemovePoints(y : ArrayLike, removeHow : str = 'absfar', p : float = 0.1, removeOrSaturate : str = 'remove') -> dict:
#     """
#     DN_RemovePoints: How time-series properties change as points are removed.

#     A proportion, p, of points are removed from the time series according to some
#     rule, and a set of statistics are computed before and after the change.

#     Parameters:
#     y (array-like): The input time series
#     removeHow (str): How to remove points from the time series:
#                       'absclose': those that are the closest to the mean,
#                       'absfar': those that are the furthest from the mean,
#                       'min': the lowest values,
#                       'max': the highest values,
#                       'random': at random.
#     p (float): The proportion of points to remove (default: 0.1)
#     removeOrSaturate (str): To remove points ('remove') or saturate their values ('saturate')

#     Returns:
#     dict: Statistics including the change in autocorrelation, time scales, mean, spread, and skewness.
#     """
#     y = np.asarray(y)
#     N = len(y)
    
#     if removeHow == 'absclose':
#         is_ = np.argsort(np.abs(y))[::-1]
#     elif removeHow == 'absfar':
#         is_ = np.argsort(np.abs(y))
#     elif removeHow == 'min':
#         is_ = np.argsort(y)[::-1]
#     elif removeHow == 'max':
#         is_ = np.argsort(y)
#     elif removeHow == 'random':
#         is_ = np.random.permutation(N)
#     else:
#         raise ValueError(f"Unknown method '{removeHow}'")
    
#     # Indices of points to *keep*:
#     rKeep = np.sort(is_[:round(N * (1 - p))])

#     # Indices of points to *transform*:
#     rTransform = np.setdiff1d(np.arange(N), rKeep)

#     # Do the removing/saturating to convert y -> yTransform
#     if removeOrSaturate == 'remove':
#         yTransform = y[rKeep]
#     elif removeOrSaturate == 'saturate':
#         # Saturate out the targeted points
#         if removeHow == 'max':
#             yTransform = y.copy()
#             yTransform[rTransform] = np.max(y[rKeep])
#         elif removeHow == 'min':
#             yTransform = y.copy()
#             yTransform[rTransform] = np.min(y[rKeep])
#         elif removeHow == 'absfar':
#             yTransform = y.copy()
#             yTransform[yTransform > np.max(y[rKeep])] = np.max(y[rKeep])
#             yTransform[yTransform < np.min(y[rKeep])] = np.min(y[rKeep])
#         else:
#             raise ValueError(f"Cannot 'saturate' when using '{removeHow}' method")
#     else:
#         raise ValueError(f"Unknown removOrSaturate option '{removeOrSaturate}'")
    
#     # Compute some autocorrelation properties
#     #print(yTransform)
#     n = 8
#     acf_y = AutoCorr(y, list(range(1, n+1)), 'Fourier')
#     #print(acf_y)
#     acf_yTransform = AutoCorr(yTransform, list(range(1, n+1)), 'Fourier')
#     #print(yTransform)
#     # Compute output statistics
#     out = {}

#     # Helper functions
#     f_absDiff = lambda x1, x2: np.abs(x1 - x2) # ignores the sign
#     f_ratio = lambda x1, x2: np.divide(x1, x2) # includes the sign

#     #print(FirstCrossing(yTransform, 'ac', 0, 'continuous'))
#     out['fzcacrat'] = f_ratio(FirstCrossing(yTransform, 'ac', 0, 'continuous'), 
#                               FirstCrossing(y, 'ac', 0, 'continuous'))
    
#     out['ac1rat'] = f_ratio(acf_yTransform[0], acf_y[0])
#     out['ac1diff'] = f_absDiff(acf_yTransform[0], acf_y[0])

#     out['ac2rat'] = f_ratio(acf_yTransform[1], acf_y[1])
#     out['ac2diff'] = f_absDiff(acf_yTransform[1], acf_y[1])
    
#     out['ac3rat'] = f_ratio(acf_yTransform[2], acf_y[2])
#     out['ac3diff'] = f_absDiff(acf_yTransform[2], acf_y[2])
    
#     out['sumabsacfdiff'] = np.sum(np.abs(acf_yTransform - acf_y))
#     out['mean'] = np.mean(yTransform)
#     out['median'] = np.median(yTransform)
#     out['std'] = np.std(yTransform, ddof=1)
    
#     out['skewnessrat'] = stats.skew(yTransform) / stats.skew(y)
#     # return kurtosis instead of excess kurtosis
#     out['kurtosisrat'] = stats.kurtosis(yTransform, fisher=False) / stats.kurtosis(y, fisher=False)

#     return out

def Quantile(y : ArrayLike, p : float = 0.5) -> float:
    """ 
    Calculates the quantile value at a specified proportion, p.

    Parameters:
    y (array-like): The input data vector
    p (float): The quantile proportion (default is 0.5, which is the median)

    Returns:
    float: The calculated quantile value

    Raises:
    ValueError: If p is not a number between 0 and 1
    """
    y = np.asarray(y)
    if p == 0.5:
        logger.info("Using quantile p = 0.5 (median) by default")
    
    if not isinstance(p, (int, float)) or p < 0 or p > 1:
        raise ValueError("p must specify a proportion, in (0,1)")
    
    return float(np.quantile(y, p, method = 'hazen'))

def ProportionValues(x : ArrayLike, propWhat : str = 'positive') -> float:
    """
    Calculate the proportion of values meeting specific conditions in a time series.

    Parameters
    ----------
    x : array-like
        Input time series or data vector
    propWhat : str, optional (default is 'positive')
        Type of values to count:
        - 'zeros': values equal to zero
        - 'positive': values strictly greater than zero
        - 'geq0': values greater than or equal to zero

    Returns
    -------
    float
        Proportion of values meeting the specified condition.
    """
    x = np.asarray(x)
    N = len(x)

    if propWhat == 'zeros':
        # returns the proportion of zeros in the input vector
        out = sum(x == 0) / N
    elif propWhat == 'positive':
        out = sum(x > 0) / N
    elif propWhat == 'geq0':
        out = sum(x >= 0) / N
    else:
        raise ValueError(f"Unknown condition to measure: {propWhat}")

    return out

def PLeft(y : ArrayLike, th : float = 0.1) -> float:
    """
    Distance from the mean at which a given proportion of data are more distant.
    
    Measures the maximum distance from the mean at which a given fixed proportion, `th`, of the time-series data points are further.
    Normalizes by the standard deviation of the time series.
    
    Parameters
    ----------
    y : array_like
        The input data vector.
    th : float, optional
        The proportion of data further than `th` from the mean (default is 0.1).
    
    Returns
    -------
    float
        The distance from the mean normalized by the standard deviation.
    """
    y = np.asarray(y)
    p = np.quantile(np.abs(y - np.mean(y)), 1-th, method='hazen')
    # A proportion, th, of the data lie further than p from the mean
    out = np.divide(p, np.std(y, ddof=1))
    return float(out)

def NLLNorm(y : ArrayLike) -> float:
    """
    Calculate the negative log-likelihood of data following a Gaussian distribution.

    This function fits a Gaussian distribution to the data using maximum likelihood
    estimation (MLE) and returns the average negative log-likelihood per data point.
    A lower value indicates the data is more likely to come from a Gaussian distribution.

    Calculated as:
        -log(L)/n = 1/2*log(2π) + log(σ) + (x - μ)²/(2σ²)
        where μ is the sample mean and σ is the sample standard deviation

    Parameters
    ----------
    y : array-like
        Input time series or data vector

    Returns
    -------
    float
        Average negative log-likelihood (NLL) per data point.
    """
    # Convert input to numpy array
    y = np.asarray(y)

    # Fit a Gaussian distribution to the data (mimicking MATLAB's normfit)
    mu = np.mean(y)
    sigma = np.std(y, ddof=1)  # ddof=1 for sample standard deviation

    # Compute the negative log-likelihood (mimicking MATLAB's normlike)
    n = len(y)
    nlogL = (n/2) * np.log(2*np.pi) + n*np.log(sigma) + np.sum((y - mu)**2) / (2*sigma**2)

    # Return the average negative log-likelihood
    return nlogL / n

def Moments(y : ArrayLike, theMom : int = 0) -> float:
    """
    A moment of the distribution of the input time series.
    Normalizes by the standard deviation.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    theMom: int, optional
        The moment to calculate. Default is 0.

    Returns
    -------
    float
        The calculated moment.
    """
    y = np.asarray(y)
    return stats.moment(y, theMom) / np.std(y, ddof=1)

def MinMax(y : ArrayLike, minOrMax : str = 'max') -> float:
    """
    The maximum and minimum values of the input data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    minOrMax : str, optional
        Return either the minimum or maximum of y. Default is 'max':
        - 'min': minimum of y
        - 'max': maximum of y

    Returns
    -------
    float
        The calculated min or max value.
    """
    y = np.asarray(y)
    if minOrMax == 'max':
        out = max(y)
    elif minOrMax == 'min':
        out = min(y)
    else:
        raise ValueError(f"Unknown method '{minOrMax}'")
    
    return out

def Mean(y : ArrayLike, meanType : str = 'arithmetic') -> float:
    """
    A given measure of location of a data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    meanType : str, optional
        Type of mean to calculate. Default is 'arithmtic':
        - 'norm' or 'arithmetic': standard arithmetic mean
        - 'median': middle value (50th percentile)
        - 'geom': geometric mean (nth root of product)
        - 'harm': harmonic mean (reciprocal of mean of reciprocals)
        - 'rms': root mean square (quadratic mean)
        - 'iqm': interquartile mean (mean of values between Q1 and Q3)
        - 'midhinge': average of first and third quartiles

    Returns
    -------
    float
        The calculated mean value.
    """
    y = np.asarray(y)
    N = len(y)

    if meanType in ['norm', 'arithmetic']:
        out = np.mean(y)
    elif meanType == 'median': # median
        out = np.median(y)
    elif meanType == 'geom': # geometric mean
        out = stats.gmean(y)
    elif meanType == 'harm': # harmonic mean
        out = N/sum(y**(-1))
    elif meanType == 'rms':
        out = np.sqrt(np.mean(y**2))
    elif meanType == 'iqm': # interquartile mean
        p = np.percentile(y, [25, 75], method='averaged_inverted_cdf')
        out = np.mean(y[(y >= p[0]) & (y <= p[1])])
    elif meanType == 'midhinge':  # average of 1st and third quartiles
        p = np.percentile(y, [25, 75], method='averaged_inverted_cdf')
        out = np.mean(p)
    else:
        raise ValueError(f"Unknown mean type '{meanType}'")

    return float(out)

def HistogramMode(y : ArrayLike, numBins : int = 10, doSimple : bool = True, doAbs : bool = False) -> float:
    """
    Measures the mode of the data vector using histograms with a given number
    of bins.

    Parameters
    -----------
    y : array-like
        the input data vector
    numBins : int, optional
        the number of bins to use in the histogram
    doSimple : bool, optional
        whether to use a simple binning method (linearly spaced bins)
    doAbs: bool, optional
        whether to take the absolute value first

    Returns
    --------
    float
        the mode of the data vector using histograms with numBins bins. 
    """
    y = np.asarray(y)
    if doAbs:
        y = np.abs(y)
    N = 0
    if doSimple:
        N, binEdges = simple_binner(y, numBins)
    else:
        binEdges = binpicker(y.min(), y.max(), numBins)
        N = histc(y, binEdges)[:-1]
    # compute bin centers from bin edges
    binCenters = np.mean([binEdges[:-1], binEdges[1:]], axis=0)

    # mean position of maximums (if multiple)
    out = np.mean(binCenters[N == np.max(N)])

    return float(out)

def HistogramAsymmetry(y : ArrayLike, numBins : int = 10, doSimple : bool = True) -> Dict[str, float]:
    """
    Calculate measures of histogram asymmetry for a time series.

    Computes various measures of asymmetry by analyzing the positive and negative 
    values in the histogram distribution separately.

    Parameters
    ----------
    y : array-like
        Input time series (1D)
    numBins : int, optional
        Number of bins to use in histogram calculation. Default is 10
    doSimple : bool, optional
        If True, uses linearly spaced bins. If False, uses optimized bin edges.

    Returns
    -------
    dict
        Dictionary containing asymmetry measures:
    """
    # check that the data is zscored
    y = np.asarray(y)
    # compute the histogram seperately from positive and negative values in the data
    yPos = y[y > 0] # filter out the positive vals
    yNeg = y[y < 0] # filter out the negative vals

    if doSimple:
        countsPos, binEdgesPos = simple_binner(yPos, numBins)
        countsNeg, binEdgesNeg = simple_binner(yNeg, numBins)
    else:
        binEdgesPos = binpicker(yPos.min(), yPos.max(), numBins)
        countsPos = histc(yPos, binEdgesPos)[:-1]
        binEdgesNeg = binpicker(yNeg.min(), yNeg.max(), numBins)
        countsNeg = histc(yNeg, binEdgesNeg)[:-1]
    # normalise by the total counts
    NnonZero = np.sum(y!=0)
    pPos = np.divide(countsPos, NnonZero)
    pNeg = np.divide(countsNeg, NnonZero)

    # compute bin centers from bin edges
    binCentersPos = np.mean([binEdgesPos[:-1], binEdgesPos[1:]], axis=0)
    binCentersNeg = np.mean([binEdgesNeg[:-1], binEdgesNeg[1:]], axis=0)

    # Histogram counts and overall density differences
    out = {}
    out['densityDiff'] = np.sum(y > 0) - np.sum(y < 0)  # measure of asymmetry about the mean
    out['modeProbPos'] = np.max(pPos)
    out['modeProbNeg'] = np.max(pNeg)
    out['modeDiff'] = out['modeProbPos'] - out['modeProbNeg']

    # Mean position of maximums (if multiple)
    out['posMode'] = np.mean(binCentersPos[pPos == out['modeProbPos']])
    out['negMode'] = np.mean(binCentersNeg[pNeg == out['modeProbNeg']])
    out['modeAsymmetry'] = out['posMode'] + out['negMode']

    return out

def HighLowMu(y: ArrayLike) -> float:
    """
    The highlowmu statistic.

    The highlowmu statistic is the ratio of the mean of the data that is above the
    (global) mean compared to the mean of the data that is below the global mean.

    Paramters
    ----------
    y (array-like): The input data vector

    Returns
    --------
    float
        The highlowmu statistic.
    """
    y = np.asarray(y)
    mu = np.mean(y) # mean of data
    mhi = np.mean(y[y > mu]) # mean of data above the mean
    mlo = np.mean(y[y < mu]) # mean of data below the mean
    out = np.divide((mhi-mu), (mu-mlo)) # ratio of the differences

    return out

def FitMLE(y : ArrayLike, fitWhat : str = 'gaussian') -> Union[Dict[str, float], float]:
    """
    Maximum likelihood distribution fit to data.

    Fits a specified probability distribution to the data using maximum likelihood 
    estimation (MLE) and returns the fitted parameters.

    Parameters
    ----------
    y : array-like
        Input time series or data vector
    fitWhat : {'gaussian', 'uniform', 'geometric'}, optional
        Distribution type to fit:
        - 'gaussian': Normal distribution (returns mean and std)
        - 'uniform': Uniform distribution (returns bounds a and b)
        - 'geometric': Geometric distribution (returns p parameter)
        Default is 'gaussian'

    Returns
    -------
    Union[Dict[str, float], float]
        For 'gaussian':
            dict with keys:
                - 'mean': location parameter
                - 'std': scale parameter
        For 'uniform':
            dict with keys:
                - 'a': lower bound
                - 'b': upper bound
        For 'geometric':
            float: success probability p
    """
    y = np.asarray(y)
    out = {}
    if fitWhat == 'gaussian':
        loc, scale = stats.norm.fit(y, method="MLE")
        out['mean'] = loc
        out['std'] = scale
    elif fitWhat == 'uniform':
        loc, scale = stats.uniform.fit(y, method="MLE")
        out['a'] = loc
        out['b'] = loc + scale 
    elif fitWhat == 'geometric':
        sampMean = np.mean(y)
        p = 1/(1+sampMean)
        return p
    else:
        raise ValueError(f"Invalid fit specifier, {fitWhat}")

    return out

def CV(x : ArrayLike, k : int = 1) -> float:
    """
    Calculate the coefficient of variation of order k.

    The coefficient of variation of order k is (sigma/mu)^k, where sigma is the
    standard deviation and mu is the mean of the input data vector.

    Parameters
    ----------
    x : array-like
        Input time series or data vector
    k : int, optional
        Order of the coefficient of variation. Default is 1.

    Returns
    -------
    float
        The coefficient of variation of order k.
    """
    if not isinstance(k, int) or k < 0:
        warnings.warn('k should probably be a positive integer')
        # carry on with just this warning, though
    
    # Compute the coefficient of variation (of order k) of the data
    return (np.std(x, ddof=1) ** k) / (np.mean(x) ** k)

def CustomSkewness(y : ArrayLike, whatSkew : str = 'pearson') -> float:
    """
    Calculate custom skewness measures of a time series.

    Computes either the Pearson or Bowley skewness. The Pearson skewness uses mean, 
    median and standard deviation, while the Bowley skewness (also known as quartile 
    skewness) uses quartiles.

    Parameters
    ----------
    y : array-like
        Input time series
    whatSkew : {'pearson', 'bowley'}, optional
        The skewness measure to calculate:
        - 'pearson': (3 * mean - median) / std
        - 'bowley': (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
        Default is 'pearson'.

    Returns
    -------
    float
        The calculated skewness measure:
        - Positive values indicate right skew
        - Negative values indicate left skew
        - Zero indicates symmetry
    """

    if whatSkew == 'pearson':
        out = ((3 * np.mean(y) - np.median(y)) / np.std(y, ddof=1))
    elif whatSkew == 'bowley':
        qs = np.quantile(y, [0.25, 0.5, 0.75], method='hazen')
        out = (qs[2]+qs[0] - 2 * qs[1]) / (qs[2] - qs[0]) 
    
    return out

def Cumulants(y: ArrayLike, cumWhatMay: str = 'skew1') -> float:
    """
    Calculate distributional moments (skewness and kurtosis) of a time series.

    This function computes higher-order moments of the input time series using
    scipy.stats functions, with options for bias correction.

    Parameters
    ----------
    y : array-like
        Input time series.
    cumWhatMay : {'skew1', 'skew2', 'kurt1', 'kurt2'}, optional
        Type of moment to calculate:
        - 'skew1': Skewness with bias
        - 'skew2': Skewness with bias correction
        - 'kurt1': Kurtosis with bias
        - 'kurt2': Kurtosis with bias correction
        Default is 'skew1'.

    Returns
    -------
    float
        The calculated moment value.
    """
    if cumWhatMay == 'skew1':
        out = stats.skew(y, bias=True)
    elif cumWhatMay == 'skew2':
        out = stats.skew(y, bias=False)
    elif cumWhatMay == 'kurt1':
        out = stats.kurtosis(y, fisher=False, bias=True)
    elif cumWhatMay == 'kurt2':
        out = stats.kurtosis(y, bias=False, fisher=False)
    else:
        raise ValueError("cumWhatMay must be one of: 'skew1', 'skew2', 'kurt1', or 'kurt2'")
    
    return out

def Burstiness(y: ArrayLike) -> Dict[str, float]:
    """
    Calculate burstiness statistics of a time series.
    
    Implements both the original Goh & Barabasi burstiness and
    the improved Kim & Jo version for finite time series.
    
    Parameters
    ----------
    y : array-like
        Input time series
    
    Returns
    -------
    dict:
        'B': Original burstiness statistic
        'B_Kim': Improved burstiness for finite series
    
    References
    ----------
    - Goh & Barabasi (2008). Europhys. Lett. 81, 48002
    - Kim & Jo (2016). http://arxiv.org/pdf/1604.01125v1.pdf
    """
    y = np.asarray(y)
    mean = np.mean(y)
    std = np.std(y, ddof=1)

    r = np.divide(std,mean) # coefficient of variation
    B = np.divide((r - 1), (r + 1)) # Original Goh and Barabasi burstiness statistic, B

    # improved burstiness statistic, accounting for scaling for finite time series
    # Kim and Jo, 2016, http://arxiv.org/pdf/1604.01125v1.pdf
    N = len(y)
    p1 = np.sqrt(N+1)*r - np.sqrt(N-1)
    p2 = (np.sqrt(N+1)-2)*r + np.sqrt(N-1)

    B_Kim = np.divide(p1, p2)

    out = {'B': B, 'B_Kim': B_Kim}

    return out
