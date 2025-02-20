import numpy as np
from scipy.interpolate import make_lsq_spline
from PeripheryFunctions.BF_iszscored import BF_iszscored
from loguru import logger

def PD_PeriodicityWang(y : list) -> np.ndarray:
    """
    Periodicity extraction measure of Wang et al. (2007)
    Implements an idea based on the periodicity extraction measure proposed in:
    "Structure-based Statistical Features and Multivariate Time Series Clustering"
    X. Wang and A. Wirth and L. Wang
    Seventh IEEE International Conference on Data Mining, 351--360 (2007)
    DOI: 10.1109/ICDM.2007.103
    
    Detrends the time series using a three-knot cubic regression spline
    and then computes autocorrelations up to one third of the length of
    the time series.
    The frequency is the first peak in the autocorrelation function satisfying
    a set of conditions.

    Note: due to indexing differences, each python output will be one less than MATLAB output.
    """
    # check time series is z-scored
    if not BF_iszscored(y):
        logger.warning("The input time series should be z-scored.")
    N = len(y)
    xdata = np.arange(0, N)
    ths = [0, 0.01,0.1,0.2,1/np.sqrt(N),5/np.sqrt(N),10/np.sqrt(N)]
    numThresholds = len(ths)
    # detrend using a regression spline with 3 knots
    numPolyPieces = 2 # number of polynomial pieces in the spline
    breaks = np.linspace(xdata[0], xdata[-1], numPolyPieces)
    splineOrder = 4 # order of the spline
    t = np.concatenate((
        np.full(splineOrder+1, breaks[0]),
        breaks[0:-1],
        np.full(splineOrder+1, breaks[-1])
    ))
    spline = make_lsq_spline(x=xdata, y=y, t=t, k=splineOrder)
    y_spl = spline(np.arange(0, N))
    y = y - y_spl

    # 2. Compute autocorrelations up to 1/3 the length of the time series.
    acmax = int(np.ceil(N/3)) # compute the autocorrelation up to this lag
    acf = np.zeros(acmax)
    for i in range(0, acmax):
        acf[i] = np.mean(y[:N-i-1] * y[i+1:N+1])
    
    # 3. Frequency is the first peak satisfying the following conditions:
    diffac = np.diff(acf) # % differenced time series
    sgndiffac = np.sign(diffac)
    bath = np.diff(sgndiffac)
    troughs = np.argwhere(bath == 2).flatten() + 1 # % finds troughs
    peaks = np.argwhere(bath == -2).flatten() + 1 # % finds peaks
    numPeaks = len(peaks)

    theFreqs = np.zeros(numThresholds)
    for k in range(numThresholds):
        theFreqs[k] = 1
        for i in range(numPeaks):
            ipeak = peaks[i] # index
            thepeak = acf[ipeak] # get the acf at the peak
            ftrough = np.argwhere(troughs < ipeak).flatten()[-1]
            if ftrough.size == 0:
                continue  # no trough found before ipeak
            itrough = troughs[ftrough]
            theTrough = acf[itrough]
    
            if thepeak - theTrough < ths[k]:
                continue

            if thepeak < 0:
                continue
            
            theFreqs[k] = ipeak
            break 

    return theFreqs
