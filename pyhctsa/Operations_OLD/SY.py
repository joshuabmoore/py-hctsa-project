import numpy as np
from typing import Union
import warnings
from CO import FirstCrossing, AutoCorr
from EN import SampEn, ApEN
from loguru import logger
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import kpss
from scipy.signal import detrend
from statsmodels.tools.sm_exceptions import InterpolationWarning
from BF_iszscored import BF_iszscored
from PN_sampenc import PN_sampenc
from DN import Moments
# temporarily turn off warnings for the test statistic being too big or small
warnings.simplefilter('ignore', InterpolationWarning)

def Trend(y : list):
    """
    Quantifies various measures of trend in a time series.

    Linearly detrends the time series using detrend, and returns the ratio of
    standard deviations before and after the linear detrending. If a strong linear
    trend is present in the time series, this operation should output a low value.
    Also fits a line and gives parameters from that fit, as well as statistics on
    a cumulative sum of the time series.

    Parameters:
    -----------
    y : array-like
        the input time series
    
    Returns:
    --------
    out : dict
        a dictionary of various measures of trend in the time series
    """
    if not BF_iszscored(y):
        warnings.warn('The input time series should be z-scored')
    
    N = len(y)

    # ratio of std before and after linear detrending
    out = {}
    dt_y = detrend(y)
    out['stdRatio'] = np.std(dt_y, ddof=1) / np.std(y, ddof=1)
    
    # do a linear fit
    # need to use the same xrange as MATLAB with 1 indexing for correct result
    coeffs = np.polyfit(range(1, N+1), y, 1)
    out['gradient'] = coeffs[0]
    out['intercept'] = coeffs[1]

    # Stats on the cumulative sum
    yC = np.cumsum(y)
    out['meanYC'] = np.mean(yC)
    out['stdYC'] = np.std(yC, ddof=1)
    coeffs_yC = np.polyfit(range(1, N+1), yC, 1)
    out['gradientYC'] = coeffs_yC[0]
    out['interceptYC'] = coeffs_yC[1]

    # Mean cumsum in first and second half of the time series
    out['meanYC12'] = np.mean(yC[:int(np.floor(N/2))])
    out['meanYC22'] = np.mean(yC[int(np.floor(N/2)):])

    return out

def StdNthDerChange(y : list, maxd : int = 10):
    """
    How the output of SY_StdNthDer changes with order parameter.
    Order parameter controls the derivative of the signal.

    Operation inspired by a comment on the Matlab Central forum: "You can
    measure the standard deviation of the n-th derivative, if you like." --
    Vladimir Vassilevsky, DSP and Mixed Signal Design Consultant from
    http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters:
    -----------
    y : array-like
        the input time series
    maxd : int, optional
        the maximum derivative to take.

    Returns:
    --------
    out : dict
        the parameters and quality of fit for an exponential model of the variation 
        across successive derivatives

    Note: Uses degree of freedom adjusted RMSE and R2 to align with MATLAB implementation.
    """
    ms = np.array([StdNthDer(y, i) for i in range(1, maxd + 1)])
    # fit exponential growth/decay
    # seed the starting point for params a, b
    p0 = [1, 0.5*np.sign(ms[-1]-ms[0])]
    expFunc = lambda x, a, b : a * np.exp(b*x)
    # fit function using nonlinear least squares
    popt, _ = curve_fit(expFunc, xdata=range(1, maxd+1), ydata=ms, p0=p0, method='lm')
    a, b = popt
    out = {}
    out['fexp_a'] = a 
    out['fexp_b'] = b
    ms_pred = expFunc(range(1, maxd+1), *popt)
    res = ms - ms_pred
    ss_res = np.sum(res**2)
    ss_tot = np.sum((ms - np.mean(ms))**2)
    r_sq = 1 - (ss_res/ss_tot)
    out['fexp_r2'] = r_sq
    out['fexp_adjr2'] = 1 - ((1-r_sq) * (len(ms)-1)) / (len(ms)-len(popt)) # d.o.f adjusted coeff of determination
    # Not mentioned in MATLAB's fitting function that RMSE is actually d.o.f adjusted. Very silly. 
    out['fexp_rmse'] = np.sqrt(np.sum(res**2)/(len(ms)-len(popt)))

    return out

def StdNthDer(y : list, ndr : int = 2):
    """
    Standard deviation of the nth derivative of the time series.

    Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    Consultant in a Matlab forum, who stated that You can measure the standard
    deviation of the nth derivative, if you like".
    cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters:
    -----------
    y : array-like
        the input time series
    n : int, optional
        the order of derivative to analyse

    Returns:
    --------
    out : float
        the std of the nth derivative of the time series
    """

    # crude method of taking a derivative that could be improved upon in future...
    yd = np.diff(y, n=ndr)
    if len(yd) == 0:
        raise ValueError(f"Time series (N = {len(y)}) too short to compute differences at n = {n}")
    out = np.std(yd, ddof=1)

    return out

def StatAv(y : list, whatType : str = 'seg', extraParam : int = 5):
    """
    Simple mean-stationarity metric.

    The StatAv measure divides the time series into non-overlapping subsegments,
    calculates the mean in each of these segments and returns the standard deviation
    of this set of means.

    Empirically mean-stationary data would display StatAv approaching to zero.

    Args:
    y (array-like): 
        The input time series
    whatType (str): The type of StatAv to perform:
        'seg': divide the time series into n segments (default)
        'len': divide the time series into segments of length n
    n (int): 
        Either the number of subsegments ('seg') (default : 5) or their length ('len').

    Returns:
    out: float 
        The StatAv statistic
    """
    N = len(y)

    if whatType == 'seg':
        # divide time series into n segments
        p = int(np.floor(N / extraParam))  # integer division, lose the last N mod n data points
        M = np.array([np.mean(y[p*j:p*(j+1)]) for j in range(extraParam)])
    elif whatType == 'len':
        if N > 2*extraParam:
            pn = int(np.floor(N / extraParam))
            M = np.array([np.mean(y[j*extraParam:(j+1)*extraParam]) for j in range(pn)])
        else:
            print(f"This time series (N = {N}) is too short for StatAv({whatType},'{extraParam}')")
            return np.nan
    else:
        raise ValueError(f"Error evaluating StatAv of type '{whatType}', please select either 'seg' or 'len'")

    s = np.std(y, ddof=1)  # should be 1 (for a z-scored time-series input)
    sdav = np.std(M, ddof=1)
    out = sdav / s

    return out 

def SpreadRandomLocal(y : list, l : Union[int, str] = 100, numSegs : int = 100, randomSeed : int = 0):
    """
    Bootstrap-based stationarity measure.
    numSegs time-series segments of length l are selected at random from the time
    series and in each segment some statistic is calculated: mean, standard
    deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1), AC(2), and the
    first zero-crossing of the autocorrelation function.
    Outputs summarize how these quantities vary in different local segments of the
    time series.

    Parameters:
    -----------
    y: array-like 
        The input time series
    l: int or str, optional
        the length of local time-series segments to analyze as a positive integer
        Can also be a specified character string:
        (i) 'ac2': twice the first zero-crossing of the autocorrelation function
        (ii) 'ac5': five times the first zero-crossing of the autocorrelation function
    numSegs: 
        the number of randomly-selected local segments to analyze
    randomSeed:
        the input to the random number generator to control reproducibility (defaults to 0)

    Returns:
    --------
    the mean and also the standard deviation of this set of 100 local estimates.

    Note: Function is very slow to compute due to reliance on the EN_SampEn function.
    """
    if isinstance(l, str):
        taug = FirstCrossing(y, 'ac', 0, 'discrete')
        if l == 'ac2':
            l = 2 * taug
        elif l == 'ac5':
            l = 5 * taug
        else:
            raise ValueError(f"Unknown specifier '{l}'")
        
        # Very short l for this sort of time series:
        if l < 5:
            print(f"Warning: This time series has a very short correlation length; "
                  f"Setting l={l} means that changes estimates will be difficult to compare...")

    N = len(y)
    if l > 0.9 * N: # operation is not suitable -- time series is too short
        warnings.warn(f"This time series (N = {N}) is too short to use l = {l}")
        return np.NaN
    
    # numSegs segments, each of length segl data points
    numFeat = 8
    qs = np.zeros((numSegs, numFeat))
    # set the random seed for reproducibility
    np.random.seed(randomSeed)

    for j in range(numSegs):
        ist = np.random.randint(N - l)
        ifh = ist + l
        ySub = y[ist:ifh]

        qs[j, 0] = np.mean(ySub)
        qs[j, 1] = np.std(ySub, ddof=1)
        qs[j, 2] = skew(ySub)
        qs[j, 3] = kurtosis(ySub)
        entropyOut = SampEn(ySub, 1, 0.15)
        qs[j, 4] = entropyOut['quadSampEn1']
        qs[j, 5] = AutoCorr(ySub, 1, 'Fourier')[0]
        qs[j, 6] = AutoCorr(ySub, 2, 'Fourier')[0]
        qs[j, 7] = FirstCrossing(ySub, 'ac', 0, 'continuous') # first zero crossing
    
    fs = np.zeros((numFeat, 2))
    fs[:, 0] = np.nanmean(qs, axis=0)
    fs[:, 1] = np.nanstd(qs, axis=0, ddof=1)

    out = {
        'meanmean': fs[0, 0], 'meanstd': fs[1, 0], 'meanskew': fs[2, 0], 'meankurt': fs[3, 0],
        'meansampen1_015': fs[4, 0], 'meanac1': fs[5, 0], 'meanac2': fs[6, 0], 'meantaul': fs[7, 0],
        'stdmean': fs[0, 1], 'stdstd': fs[1, 1], 'stdskew': fs[2, 1], 'stdkurt': fs[3, 1],
        'stdsampen1_015': fs[4, 1], 'stdac1': fs[5, 1], 'stdac2': fs[6, 1], 'stdtaul': fs[7, 1]
    }

    return out

def SlidingWindow(y : list, windowStat : str = 'mean', acrossWinStat : str = 'std', numSeg : int = 5, incMove : int = 2) -> dict:
    """
    Sliding window measures of stationarity.
    This function is based on sliding a window along the time series, measuring
    some quantity in each window, and outputting some summary of this set of local
    estimates of that quantity.
    Another way of saying it: calculate 'windowStat' in each window, and computes
    'acrossWinStat' for the set of statistics calculated in each window.
    lillie, ent no yet implemented as windowStats
    """

    winLength = np.floor(len(y)/numSeg)
    if winLength == 0:
        warnings.warn(f"Time-series of length {len(y)} is too short for {numSeg} windows")
        return np.nan
    inc = np.floor(winLength/incMove) # increment to move at each step
    # if incrment rounded down to zero, prop it up 
    if inc == 0:
        inc = 1
    
    numSteps = int(np.floor((len(y)-winLength)/inc) + 1)
    qs = np.zeros(numSteps)
    
    # convert a step index (stepInd) to a range of indices corresponding to that window
    def get_window(stepInd: int):
        start_idx = (stepInd) * inc
        end_idx = (stepInd) * inc + winLength
        return np.arange(start_idx, end_idx).astype(int)

    if windowStat == 'mean':
        for i in range(numSteps):
            qs[i] = np.mean(y[get_window(i)])
    elif windowStat == 'std':
        for i in range(numSteps):
            qs[i] = np.std(y[get_window(i)], ddof=1)
    elif windowStat == 'ent':
        warnings.warn(f"{windowStat} not yet implemented")
    elif windowStat == 'apen':
        for i in range(numSteps):
            qs[i] = ApEN(y[get_window(i)], 1, 0.2)
    elif windowStat == 'sampen':
        for i in range(numSteps):
            sampen_dict = SampEn(y[get_window(i)], 1, 0.1)
            qs[i] = sampen_dict['sampen1']
    elif windowStat == 'mom3':
        for i in range(numSteps):
            qs[i] = Moments(y[get_window(i)], 3)
    elif windowStat == 'mom4':
        for i in range(numSteps):
            qs[i] = Moments(y[get_window(i)], 4)
    elif windowStat == 'mom5':
        for i in range(numSteps):
            qs[i] = Moments(y[get_window(i)], 5)
    elif windowStat == 'AC1':
        for i in range(numSteps):
            qs[i] = AutoCorr(y[get_window(i)], 1, 'Fourier')
    elif windowStat == 'lillie':
        warnings.warn(f"{windowStat} not yet implemented")
    else:
        raise ValueError(f"Unknown statistic '{windowStat}'")
    

    if acrossWinStat == 'std':
        out = np.std(qs, ddof=1)/np.std(y, ddof=1)
    elif acrossWinStat == 'apen':
        out = ApEN(qs, 1, 0.2)
    elif acrossWinStat == 'sampen':
        sampen_dict = SampEn(qs, 2, 0.15)
        out = sampen_dict['quadSampEn1']
    elif acrossWinStat == 'ent':
        warnings.warn(f"{acrossWinStat} not yet implemented")
        out = np.nan
    else:
        raise ValueError(f"Unknown statistic '{acrossWinStat}'")
    
    return out

def RangeEvolve(y : list) -> dict:
    """
    How the time-series range changes across time.

    Measures of the range of the time series as a function of time,
    i.e., range(x_{1:i}) for i = 1, 2, ..., N, where N is the length of the time
    series.

    Parameters:
    y : array-like
        The input time series

    Returns:
    out : dict 
        A dictionary containing various metrics based on the dynamics of how new extreme events occur with time.
    """
    N = len(y)
    out = {} # initialise storage
    cums = np.zeros(N)
    for i in range(N):
        cums[i] = np.ptp(y[:i+1])  # np.ptp calculates the range (peak to peak)
    
    fullr = np.ptp(y)

    # return number of unqiue entries in a vector, x
    lunique = lambda x : len(np.unique(x))
    out['totnuq'] = lunique(cums)

    # how many of the unique extrema are in the first <proportions> of time series? 
    cumtox = lambda x : lunique(cums[:int(np.floor(N*x))])/out['totnuq']
    out['nuqp1'] = cumtox(0.01)
    out['nuqp10'] = cumtox(0.1)
    out['nuqp20'] = cumtox(0.2)
    out['nuqp50'] = cumtox(0.5)

    # how many unique extrema are in the first <length> of time series? 
    Ns = [10, 50, 100, 1000]
    for Nval in Ns:
        if N >= Nval:
            out[f'nuql{Nval}'] = lunique(cums[:Nval])/out['totnuq']
        else:
            out[f'nuql{N}'] = np.NaN
    
    # (**2**) Actual proportion of full range captured at different points
    out['p1'] = cums[int(np.ceil(N*0.01))]/fullr
    out['p10'] = cums[int(np.ceil(N*0.1))]/fullr
    out['p20'] = cums[int(np.ceil(N*0.2))]/fullr
    out['p50'] = cums[int(np.ceil(N*0.5))]/fullr

    for Nval in Ns:
        if N >= Nval:
            out[f'l{Nval}'] = cums[Nval-1]/fullr
        else:
            out[f'l{Nval}'] = np.NaN

    return out

def LocalGlobal(y : list, subsetHow : str = 'l', nsamps : Union[int, float, None] = None, randomSeed : int = 0) -> dict:
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
    if not BF_iszscored(y):
        logger.warning(f"The input time series should be z-scored")
    
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
    out['sampen101'] = PN_sampenc(y[r], 1, 0.1, True)[0][0]/PN_sampenc(y, 1, 0.1, True)[0][0]

    return out

def KPSStest(y : list, lags : Union[int, list] = 0) -> dict:
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

def DynWin(y : list, maxNumSegments : int = 10):
    """
    How stationarity estimates depend on the number of time-series subsegments.
    
    Specifically, variation in a range of local measures are implemented: mean,
    standard deviation, skewness, kurtosis, ApEn(1,0.2), SampEn(1,0.2), AC(1),
    AC(2), and the first zero-crossing of the autocorrelation function.
    
    The standard deviation of local estimates of these quantities across the time
    series are calculated as an estimate of the stationarity in this quantity as a
    function of the number of splits, n_{seg}, of the time series.

    Parameters:
    -----------
    y : array_like
        the time series to analyze.
    maxNumSegments : int, optional
        the maximum number of segments to consider. Sweeps from 2 to
        maxNumSegments. Defaults to 10. 
    
    Returns:
    --------
    out : dict
        the standard deviation of this set of 'stationarity' estimates across these window sizes
    """
    nsegr = np.arange(2, maxNumSegments+1, 1) # range of nseg to sweep across
    nmov = 1 # controls window overlap
    numFeatures = 11 # num of features
    fs = np.zeros((len(nsegr), numFeatures)) # standard deviation of feature values over windows
    taug = FirstCrossing(y, 'ac', 0, 'discrete') # global tau

    for i, nseg in enumerate(nsegr):
        wlen = int(np.floor(len(y)/nseg)) # window length
        inc = int(np.floor(wlen/nmov)) # increment to move at each step
        # if increment is rounded to zero, prop it up
        if inc == 0:
            inc = 1
        
        numSteps = int(np.floor((len(y) - wlen)/inc) + 1)
        qs = np.zeros((numSteps, numFeatures))

        for j in range(numSteps):
            ySub = y[j*inc:j*inc+wlen]
            taul = FirstCrossing(ySub, 'ac', 0, 'discrete')

            qs[j, 0] = np.mean(ySub)
            qs[j, 1] = np.std(ySub, ddof=1)
            qs[j, 2] = skew(ySub)
            qs[j, 3] = kurtosis(ySub)
            sampenOut = SampEn(ySub, 2, 0.15)
            qs[j, 4] = sampenOut['quadSampEn1'] # SampEn_1_015
            qs[j, 5] = sampenOut['quadSampEn2'] # SampEn_2_015
            qs[j, 6] = AutoCorr(ySub, 1, 'Fourier')[0] # AC1
            qs[j, 7] = AutoCorr(ySub, 2, 'Fourier')[0] # AC2
            # (Sometimes taug or taul can be longer than ySub; then these will output NaNs:)
            qs[j, 8] = AutoCorr(ySub, taug, 'Fourier')[0] # AC_glob_tau
            qs[j, 9] = AutoCorr(ySub, taul, 'Fourier')[0] # AC_loc_tau
            qs[j, 10] = taul
        
        fs[i, :numFeatures] = np.std(qs, ddof=1, axis=0)

    # fs contains std of quantities at all different 'scales' (segment lengths)
    fs = np.std(fs, ddof=1, axis=0) # how much does the 'std stationarity' vary over different scales?

    # Outputs
    out = {}
    out['stdmean'] = fs[0]
    out['stdstd'] = fs[1]
    out['stdskew'] = fs[2]
    out['stdkurt'] = fs[3]
    out['stdsampen1_015'] = fs[4]
    out['stdsampen2_015'] = fs[5]
    out['stdac1'] = fs[6]
    out['stdac2'] = fs[7]
    out['stdactaug'] = fs[8]
    out['stdactaul'] = fs[9]
    out['stdtaul'] = fs[10]

    return out 

def DriftingMean(y : list, segmentHow : str = 'num', l : Union[None, int] = None):
    """
    Mean and variance in local time-series subsegments.
    Splits the time series into segments, computes the mean and variance in each
    segment and compares the maximum and minimum mean to the mean variance.

    This function implements an idea found in the Matlab Central forum:
    http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters:
    -----------
    y : array-like
        the input time series
    segmentHow : str, optional
        (i) 'fix': fixed-length segments (of length l)
        (ii) 'num': a given number, l, of segments
    l : int, optional
        either the length ('fix') or number of segments ('num')
    
    Returns:
    --------
    out : dict
        dictionary of statistics pertaining to mean and variance in local time-series subsegments
    """
    N = len(y)
    
    if l is None:
        if segmentHow == 'num':
            l = 5 # 5 segments
        elif segmentHow == 'fix':
            l = 200 # 200 sample segments

    if segmentHow == 'num':
        l = int(np.floor(N/l))
    elif segmentHow != 'fix':
        raise ValueError(f"Unknown input setting {segmentHow}")
    
    # Check for short time series
    if l == 0 or N < l: # doesn't make sense to split into more windows than there are data points
        return np.NaN
    
    # get going
    numFits = int(np.floor(N/l)) # number of times l fits completely into N
    z = np.zeros((l, numFits))
    for i in range(numFits):
        z[:, i] = y[i*l : (i+1)*l]
    zm = np.mean(z, axis=0)
    zv = np.var(z, ddof=1, axis=0)
    meanVar = np.mean(zv)
    maxMean = np.max(zm)
    minMean = np.min(zm)
    meanMean = np.mean(zm)

    # Output stats
    out = {}
    out['max'] = maxMean/meanVar
    out['min'] = minMean/meanVar
    out['mean'] = meanMean/meanVar
    out['meanmaxmin'] = (out['max'] + out['min'])/2
    out['meanabsmaxmin'] = (np.abs(out['max']) + np.abs(out['min']))/2

    return out
