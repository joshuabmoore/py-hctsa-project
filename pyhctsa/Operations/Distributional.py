import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union
from scipy import stats
import warnings
from utilities import simple_binner, binpicker, histc

def Moments(y : list, theMom : int = 0) -> float:
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

def MinMax(y : ArrayLike, minOrMax : str = 'max') -> float:
    """
    The maximum and minimum values of the input data vector.

    Parameters
    ----------
    y : array-like
        Input time series or data vector

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

