import numpy as np
from scipy.optimize import curve_fit
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.CO_FirstCrossing import CO_FirstCrossing
from PeripheryFunctions.BF_SignChange import BF_SignChange
import warnings

def CO_AutoCorrShape(y, stopWhen = 'posDrown'):
    """
    CO_AutoCorrShape: How the autocorrelation function changes with the time lag.

    Outputs include the number of peaks, and autocorrelation in the
    autocorrelation function (ACF) itself.

    Parameters:
    -----------
    y : array_like
        The input time series
    stopWhen : str or int, optional
        The criterion for the maximum lag to measure the ACF up to.
        Default is 'posDrown'.

    Returns:
    --------
    dict
        A dictionary containing various metrics about the autocorrelation function.
    """
    N = len(y)

    # Only look up to when two consecutive values are under the significance threshold
    th = 2 / np.sqrt(N)  # significance threshold

    # Calculate the autocorrelation function, up to a maximum lag, length of time series (hopefully it's cropped by then)
    acf = []

    # At what lag does the acf drop to zero, Ndrown (by my definition)?
    if isinstance(stopWhen, int):
        taus = list(range(0, stopWhen+1))
        acf = CO_AutoCorr(y, taus, 'Fourier')
        Ndrown = stopWhen
        
    elif stopWhen in ['posDrown', 'drown', 'doubleDrown']:
        # Compute ACF up to a given threshold:
        Ndrown = 0 # the point at which ACF ~ 0
        if stopWhen == 'posDrown':
            # stop when ACF drops below threshold, th
            for i in range(1, N+1):
                acf_val = CO_AutoCorr(y, i-1, 'Fourier')[0]
                if np.isnan(acf_val):
                    warnings.warn("Weird time series (constant?)")
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
                acf_val = CO_AutoCorr(y, i-1, 'Fourier')[0] # acf vector indicies are not lags
                # if positive and less than thresh
                if i > 0 and abs(acf_val) < th:
                    Ndrown = i
                    acf.append(acf_val)
                    break
                acf.append(acf_val)
        elif stopWhen == 'doubleDrown':
            # Stop at 2*tau, where tau is the lag where ACF ~ 0 (within 1/sqrt(N) threshold)
            for i in range(1, N+1):
                acf_val = CO_AutoCorr(y, i-1, 'Fourier')[0]
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
        out = np.NaN
    
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
        out['ac1'] = CO_AutoCorr(acf, 1, 'Fourier')[0]
        if all(acf > 0):
            out['actau'] = np.nan
        else:
            out['actau'] = CO_AutoCorr(acf, CO_FirstCrossing(acf, 'ac', 0, 'discrete'), 'Fourier')[0]

    else:
        out['ac1'] = np.nan
        out['actau'] = np.nan
    
    # Local extrema
    dacf = np.diff(acf)
    ddacf = np.diff(dacf)
    extrr = BF_SignChange(dacf, 1)
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
        out['fexpacf_stdres'] = np.std(residuals2, ddof=1) # IMPORTANT *** DDOF=1 TO MATCH MATLAB STD ***

    else:
        # Fit inappropriate (or failed): return NaNs for the relevant stats
        out['decayTimescale'] = np.nan
        out['fexpacf_r2'] = np.nan
        out['fexpacf_stdres'] = np.nan
    
    return out
