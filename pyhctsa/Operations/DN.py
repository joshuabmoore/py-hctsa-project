import numpy as np
from scipy import stats
from scipy.stats import moment
from loguru import logger
import warnings
from typing import Union
from CO import AutoCorr, FirstCrossing
# Replace the following with local versions eventually...
from BF_SimpleBinner import BF_SimpleBinner
from BF_iszscored import BF_iszscored

def Withinp(x : list, p : Union[float, int] = 1.0, meanOrMedian : str = 'mean') -> float:
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
    return np.sum((x >= mu - p * sig) & (x <= mu + p * sig)) / N

def Unique(x : list) -> float:
    """
    The proportion of the time series that are unique values.

    Parameters:
    x (array-like): the input data vector

    Returns:
    out (float): the proportion of time series that are unique values
    """
    out = len(np.unique(x))/len(x)

    return out

def TrimmedMean(y : list, p_exclude : float = 0.0) -> float:
    """
    Mean of the trimmed time series using trimmean.

    Parameters:
    ----------
    y (array-like): the input time series
    n (float): the percentage of highest and lowest values in y to exclude from the mean calculation

    Returns:
    --------
    out (float): the mean of the trimmed time series.
    """
    p_exclude *= 0.01
    N = len(y)
    trim = int(np.round(N * p_exclude / 2))
    y = np.sort(y)

    out = np.mean(y[trim:N-trim])

    return out

def Spread(y : list, spreadMeasure : str = 'std') -> float:
    """
    Measure of spread of the input time series.
    Returns the spread of the raw data vector, as the standard deviation,
    inter-quartile range, mean absolute deviation, or median absolute deviation.
    """
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

def RemovePoints(y : list, removeHow : str = 'absfar', p : float = 0.1, removeOrSaturate : str = 'remove') -> dict:
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
        logger.warning("The input time series should be z-scored.")
    
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
    n = 8
    acf_y = AutoCorr(y, list(range(1, n+1)), 'Fourier')
    acf_yTransform = AutoCorr(yTransform, list(range(1, n+1)), 'Fourier')

    # Compute output statistics
    out = {}

    # Helper functions
    f_absDiff = lambda x1, x2: abs(x1 - x2) # ignores the sign
    f_ratio = lambda x1, x2: x1 / x2 # includes the sign

    out['fzcacrat'] = f_ratio(FirstCrossing(yTransform, 'ac', 0, 'continuous'), 
                              FirstCrossing(y, 'ac', 0, 'continuous'))
    
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

def Quantile(y : list, p : float = 0.5) -> float:
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
    if p == 0.5:
        logger.info("Using quantile p = 0.5 (median) by default")
    
    if not isinstance(p, (int, float)) or p < 0 or p > 1:
        raise ValueError("p must specify a proportion, in (0,1)")
    
    return np.quantile(y, p, method = 'hazen')

def ProportionValues(x : list, propWhat : str = 'positive'):
    """
    Proportion of values in a data vector.
    Returns statistics on the values of the data vector: the proportion of zeros,
    the proportion of positive values, and the proportion of values greater than or
    equal to zero.

    Parameters:
    x (array-like): the input time series
    propWhat (str): the proportion of a given type of value in the time series: 
        (i) 'zeros': values that equal zero
        (ii) 'positive': values that are strictly positive
        (iii) 'geq0': values that are greater than or equal to zero

    Returns:
    out (float) : proportion of given type of value
    """

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

def Pleft(y : list, th : float = 0.1):
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
    p = np.quantile(np.abs(y - np.mean(y)), 1-th, method='hazen')

    # A proportion, th, of the data lie further than p from the mean
    out = p/np.std(y, ddof=1)

    return out

def nlogL_norm(y : list):
    """
    Negative log likelihood of data coming from a Gaussian distribution.

    This function fits a Gaussian distribution to the data and returns the negative
    log likelihood of the data coming from that Gaussian distribution.

    Parameters:
    y (array-like): A vector of data

    Returns:
    float: The negative log likelihood per data point
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

def Moments(y : list, theMom : int = 0):
    """
    A moment of the distribution of the input time series.
    Normalizes by the standard deviation.

    Parameters:
    y (array-like): the input data vector
    theMom (int): the moment to calculate (a scalar)

    Returns:
    out (float): theMom moment of the distribution of the input time series. 
    """
    out = moment(y, theMom) / np.std(y, ddof=1) # normalized

    return out

def MinMax(y : list, minOrMax : str = 'max'):
    """
    The maximum and minimum values of the input data vector.

    """
    if minOrMax == 'max':
        out = max(y)
    elif minOrMax == 'min':
        out = min(y)
    else:
        raise ValueError(f"Unknown method '{minOrMax}'")
    
    return out

def Mean(y : list, mean_type : str = 'arithmetic'):
    """
    A given measure of location of a data vector.

    Parameters:
    y (array-like): The input data vector
    mean_type (str): The type of mean to calculate
        'norm' or 'arithmetic': arithmetic mean
        'median': median
        'geom': geometric mean
        'harm': harmonic mean
        'rms': root-mean-square
        'iqm': interquartile mean
        'midhinge': midhinge

    Returns:
    out (float): The calculated mean value

    Raises:
    ValueError: If an unknown mean type is specified

    Notes:
    Harmonic mean only defined for positive values.
    """
    y = np.array(y)
    N = len(y)

    if mean_type in ['norm', 'arithmetic']:
        out = np.mean(y)
    elif mean_type == 'median': # median
        out = np.median(y)
    elif mean_type == 'geom': # geometric mean
        out = stats.gmean(y)
    elif mean_type == 'harm': # harmonic mean
        out = N/sum(y**(-1))
    elif mean_type == 'rms':
        out = np.sqrt(np.mean(y**2))
    elif mean_type == 'iqm': # interquartile mean, cf. DN_TrimmedMean
        p = np.percentile(y, [25, 75], method='averaged_inverted_cdf')
        out = np.mean(y[(y >= p[0]) & (y <= p[1])])
    elif mean_type == 'midhinge':  # average of 1st and third quartiles
        p = np.percentile(y, [25, 75], method='averaged_inverted_cdf') # method to match MATLAB
        out = np.mean(p)
    else:
        raise ValueError(f"Unknown mean type '{mean_type}'")

    return out

def HistogramMode(y : list, numBins : int = 10, doSimple : bool = True, doAbs : bool = False):
    """
    Mode of a data vector.
    Measures the mode of the data vector using histograms with a given number
    of bins.
    Note that when using doSoimple = False, values won't map directly onto the
    HCTSA outputs due to differences in how the bins are constructed.
    However, the trends are similar.

    Parameters:
    -----------
    y : array-like
        the input data vector
    numBins : int, optional
        the number of bins to use in the histogram
    doSimple : bool, optional
        whether to use a simple binning method (linearly spaced bins)
    doAbs: bool, optional
        whether to take the absolute value first

    Returns:
    --------
    out : float
        the mode of the data vector using histograms with numBins bins. 
    """

    if doAbs:
        y = np.abs(y)

    if isinstance(numBins, int):
        if doSimple:
            N, binEdges = BF_SimpleBinner(y, numBins)
        else:
            # this gives a different result to MATLAB for the same number of bins 
            # better to use the simple binner (as set by default)
            N, binEdges = np.histogram(y, bins=numBins)
    elif isinstance(numBins, str):
        # NOTE: auto doesn't yield the same number of bins as MATLAB's auto, despite both using the same binning algs. 
        bin_edges = np.histogram_bin_edges(y, bins=numBins)
        N, binEdges = np.histogram(y, bins=bin_edges)
    else:
        raise ValueError("Unknown format for numBins")

    # compute bin centers from bin edges
    binCenters = np.mean([binEdges[:-1], binEdges[1:]], axis=0)

    # mean position of maximums (if multiple)
    out = np.mean(binCenters[N == np.max(N)])

    return out

def HistogramAsymmetry(y : list, numBins : int = 10, doSimple : bool = True):
    """
    Measures of distributional asymmetry
    Measures the asymmetry of the histogram distribution of the input data vector.
    Note that when using `doSimple = False`, output will be different to MATLAB due 
    to differences in the histogram function.

    Parameters:
    -----------
    y : array-like
        the input data vector
    numBins : int, optional
        the number of bins to use in the histogram
    doSimple : bool, optional
        whether to use a simple binning method (linearly spaced bins)
    
    Returns:
    --------
    out : dict
        dictionary containing measures of the asymmetry of the histogram distribution

    """
    if not BF_iszscored(y):
        logger.warning("DN_HistogramAsymmetry assumes a z-scored (or standardised) input")
    
    # compute the histogram seperately from positive and negative values in the data
    yPos = y[y > 0] # filter out the positive vals
    yNeg = y[y < 0]

    if doSimple:
        countsPos, binEdgesPos = BF_SimpleBinner(yPos, numBins)
        countsNeg, binEdgesNeg = BF_SimpleBinner(yNeg, numBins)
    else:
        countsPos, binEdgesPos = np.histogram(yPos, numBins)
        countsNeg, binEdgesNeg = np.histogram(yNeg, numBins)
    
    # normalise by the total counts
    NnonZero = np.sum(y!=0)
    pPos = countsPos/NnonZero
    pNeg = countsNeg/NnonZero

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

def HighLowMu(y: list):
    """
    The highlowmu statistic.

    The highlowmu statistic is the ratio of the mean of the data that is above the
    (global) mean compared to the mean of the data that is below the global mean.

    Paramters:
    ----------
    y (array-like): The input data vector

    Returns:
    --------
    float: The highlowmu statistic.
    """
    mu = np.mean(y) # mean of data
    mhi = np.mean(y[y > mu]) # mean of data above the mean
    mlo = np.mean(y[y < mu]) # mean of data below the mean
    out = np.divide((mhi-mu), (mu-mlo)) # ratio of the differences

    return out

def Fit_mle(y : list, fitWhat : str = 'gaussian'):
    """
    Maximum likelihood distribution fit to data.
    Fits either a Gaussian, Uniform, or Geometric distribution to the data using
    maximum likelihood estimation.

    Parameters:
    -----------
    y (array-like): The input data vector
    fitWhat (str, optional): the type of fit to do
        - 'gaussian'
        - 'uniform'
        - 'geometric'
    Returns:
    --------
    dict: distirbution-specific paramters from the fit
    """

    out = {}
    if fitWhat == 'gaussian':
        loc, scale = stats.norm.fit(y, method="MLE")
        out['mean'] = loc
        out['std'] = scale
    elif fitWhat == 'uniform': # turns out to be shithouse
        loc, scale = stats.uniform.fit(y, method="MLE")
        out['a'] = loc
        out['b'] = loc + scale 
    elif fitWhat == 'geometric':
        sampMean = np.mean(y)
        p = 1/(1+sampMean)
        out['p'] = p
    else:
        raise ValueError(f"Invalid fit specifier, {fitWhat}")

    return out

def CV(x : list, k : int = 1):
    """
    Coefficient of variation

    Coefficient of variation of order k is sigma^k / mu^k (for sigma, standard
    deviation and mu, mean) of a data vector, x

    Parameters:
    ----------
    x (array-like): The input data vector
    k (int, optional): The order of coefficient of variation (k = 1 is default)

    Returns:
    -------
    float: The coefficient of variation of order k
    """
    if not isinstance(k, int) or k < 0:
        warnings.warn('k should probably be a positive integer')
        # Carry on with just this warning, though
    
    # Compute the coefficient of variation (of order k) of the data
    return (np.std(x, ddof=1) ** k) / (np.mean(x) ** k)

def CustomSkewness(y : list, whatSkew : str = 'pearson'):
    """
    Custom skewness measures
    Compute the Pearson or Bowley skewness.

    Parameters:
    -----------
    y : array_like
        Input time series
    whatSkew : str, optional
        The skewness measure to calculate:
            - 'pearson'
            - 'bowley'

    Returns:
    --------
    out : float
        The custom skewness measure.
    """

    if whatSkew == 'pearson':
        out = ((3 * np.mean(y) - np.median(y)) / np.std(y, ddof=1))
    elif whatSkew == 'bowley':
        qs = np.quantile(y, [0.25, 0.5, 0.75], method='hazen')
        out = (qs[2]+qs[0] - 2 * qs[1]) / (qs[2] - qs[0]) 
    
    return out

def Cumulants(y : list, cumWhatMay : str = 'skew1'):
    """
    Distributional moments of the input data.
    Very simple function that uses the skewness and kurtosis functions
    to calculate these higher order moments of input time series, y

    Parameters:
    ----------
    y (array-like) : the input time series
    cumWhatMay (str, optional) : the type of higher order moment
        (i) 'skew1', skewness
        (ii) 'skew2', skewness correcting for bias
        (iii) 'kurt1', kurtosis
        (iv) 'kurt2', kurtosis correcting for bias

    Returns:
    --------
    float : the higher order moment.
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
        raise ValueError('Requested Unknown cumulant must be: skew1, skew2, kurt1, or kurt2')
    
    return out

def Burstiness(y : list):
    """
    Calculate the burstiness statistic of a time series.

    This function returns the 'burstiness' statistic as defined in
    Goh and Barabasi's paper, "Burstiness and memory in complex systems,"
    Europhys. Lett. 81, 48002 (2008).

    Parameters
    ----------
    y : array-like
        The input time series.
    
    Returns
    -------
    dict
        The original burstiness statistic, B, and the improved
        burstiness statistic, B_Kim.
    """
    
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
