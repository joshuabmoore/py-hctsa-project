from PeripheryFunctions.BF_SignChange import BF_SignChange
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.IN_AutoMutualInfo import IN_AutoMutualInfo
import numpy as np
from scipy import stats

def IN_AutoMutualInfoStats(y, maxTau=None, estMethod='kernel', extraParam=None):
    """
    Statistics on automutual information function of a time series.

    Parameters:
    ----------
    y (array-like) : column vector of time series.
    estMethod (str) : input to IN_AutoMutualInfo
    extraParam (str, int, optional) : input to IN_AutoMutualInfo
    maxTau (int) : maximal time delay

    Returns:
    --------
    out (dict) : a dictionary containing statistics on the AMIs and their pattern across the range of specified time delays.
    """

    N = len(y) # length of the time series
    
    # maxTau: the maximum time delay to investigate
    if maxTau is None:
        maxTau = np.ceil(N/4)
    maxTau0 = maxTau

    # Don't go above N/2
    maxTau = min(maxTau, np.ceil(N/2))

    # Get the AMI data
    maxTau = int(maxTau)
    maxTau0 = int(maxTau0)
    timeDelay = list(range(1, maxTau+1))
    ami = IN_AutoMutualInfo(y, timeDelay=timeDelay, estMethod=estMethod, extraParam=extraParam)
    ami = np.array(list(ami.values()))

    out = {} # create dict for storing results
    # Output the raw values
    for i in range(1, maxTau0+1):
        if i <= maxTau:
            out[f'ami{i}'] = ami[i-1]
        else:
            out[f'ami{i}'] = np.nan

    # Basic statistics
    lami = len(ami)
    out['mami'] = np.mean(ami)
    out['stdami'] = np.std(ami, ddof=1)

    # First minimum of mutual information across range
    dami = np.diff(ami)
    extremai = np.where((dami[:-1] * dami[1:]) < 0)[0]
    out['pextrema'] = len(extremai) / (lami - 1)
    out['fmmi'] = min(extremai) + 1 if len(extremai) > 0 else lami

    # Look for periodicities in local maxima
    maximai = np.where((dami[:-1] > 0) & (dami[1:] < 0))[0] + 1
    dmaximai = np.diff(maximai)
    # Is there a big peak in dmaxima? (no need to normalize since a given method inputs its range; but do it anyway... ;-))
    out['pmaxima'] = len(dmaximai) / (lami // 2)
    if len(dmaximai) == 0:  # fewer than 2 local maxima
        out['modeperiodmax'] = np.nan
        out['pmodeperiodmax'] = np.nan
    else:
        out['modeperiodmax'] = stats.mode(dmaximai, keepdims=True).mode[0]
        out['pmodeperiodmax'] = np.sum(dmaximai == out['modeperiodmax']) / len(dmaximai)

    # Look for periodicities in local minima
    minimai = np.where((dami[:-1] < 0) & (dami[1:] > 0))[0] + 1
    dminimai = np.diff(minimai)

    out['pminima'] = len(dminimai) / (lami // 2)

    if len(dminimai) == 0:  # fewer than 2 local minima
        out['modeperiodmin'] = np.nan
        out['pmodeperiodmin'] = np.nan
    else:
        out['modeperiodmin'] = stats.mode(dminimai, keepdims=True).mode[0]
        out['pmodeperiodmin'] = np.sum(dminimai == out['modeperiodmin']) / len(dminimai)
    
    # Number of crossings at mean/median level, percentiles
    out['pcrossmean'] = np.mean(np.diff(np.sign(ami - np.mean(ami))) != 0)
    out['pcrossmedian'] = np.mean(np.diff(np.sign(ami - np.median(ami))) != 0)
    out['pcrossq10'] = np.mean(np.diff(np.sign(ami - np.percentile(ami, 10))) != 0)
    out['pcrossq90'] = np.mean(np.diff(np.sign(ami - np.percentile(ami, 90))) != 0)
    
    # ac1 
    out['amiac1'] = CO_AutoCorr(ami, 1, 'Fourier')[0]

    return out 
