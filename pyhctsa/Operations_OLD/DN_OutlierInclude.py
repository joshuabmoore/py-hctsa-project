import numpy as np
import warnings
from PeripheryFunctions.BF_iszscored import BF_iszscored


def DN_OutlierInclude(y, thresholdHow : str = 'abs', inc : float = 0.01):
    #% If the time series is a constant causes issues
    if np.all(y[1] == y):
        # % This method is not suitable for such time series: return a NaN
        warnings.warn("The time series is a constant!")
        return np.nan
    
    # % Check z-scored time series
    if not BF_iszscored(y):
        warnings.warn("The input time series should be z-scored")
    N = len(y)
    
    #%% Initialize thresholds
    # % ------------------------------------------------------------------------------
    # % Could be better to just use a fixed number of increments here, from 0 to the max.
    # % (rather than forcing a fixed inc)
    if thresholdHow == 'abs': # % analyze absolute value deviations
        thr = np.arange(0, max(abs(y)), inc)
        tot = N
    elif thresholdHow == 'pos':
        thr = np.arange(0, max(y), inc)
        tot = sum(y >= 0)
    elif thresholdHow == 'neg':
        thr = np.arange(0, max(-y), inc)
        tot = sum(y <= 0)
    else:
        raise ValueError(f"Error thresholding with '{thresholdHow}'. Must select either 'abs', 'pos', or 'neg'.")
    
    if len(thr) == 0:
        raise ValueError("Error setting increments through the time-series values...")
    
    msDt = np.zeros((len(thr), 6))

    # % Calculate statistics of over-threshold events, looping over thresholds
    for (i, th) in enumerate(thr):
        if thresholdHow == 'abs':
            r = np.argwhere(abs(y >= th)).flatten()
        elif thresholdHow == 'pos':
            r = np.argwhere(y >= th).flatten()
        elif thresholdHow == 'neg':
            r = np.argwhere(y <= -th).flatten()
        
        Dt_exc = np.diff(r)
        msDt[i, 0] = np.mean(Dt_exc)
        msDt[i, 1] = np.std(Dt_exc, ddof=1)/np.sqrt(len(r))
        msDt[i, 2] = len(Dt_exc)/tot*100
        msDt[i, 3] = (np.median(r)/(N/2)) - 1
        msDt[i, 4] = np.mean(r)/(N/2) - 1
        msDt[i, 5] = np.std(r, ddof=1)/np.sqrt(len(r))

    #%% Trim
    fbi = np.argmax(np.isnan(msDt[:, 0])) if np.any(np.isnan(msDt[:, 0])) else None
    if fbi:
        msDt = msDt[:fbi, :]
        thr = thr[:fbi]
    trimthr = 2 
    mj_indices = np.where(msDt[:, 2] > trimthr)[0]
    if mj_indices.size > 0:
        mj = mj_indices[-1]  # last index
        # In MATLAB: msDt = msDt(1:mj,:); we slice up to mj+1 in Python
        msDt = msDt[:mj+1, :]
        thr = thr[:mj+1]




    return fbi
