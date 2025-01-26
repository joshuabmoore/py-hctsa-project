import numpy as np
import warnings
from Operations.EN_ApEn import EN_ApEN
from Operations.DN_Moments import DN_Moments
from Operations.CO_AutoCorr import CO_AutoCorr

def SY_SlidingWindow(y : list, windowStat : str = 'mean', acrossWinStat : str = 'std', numSeg : int = 5, incMove : int = 2) -> dict:

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
            qs[i] = EN_ApEN(y[get_window(i)], 1, 0.2)
    elif windowStat == 'sampen':
        warnings.warn(f"{windowStat} not yet implemented")
    elif windowStat == 'mom3':
        for i in range(numSteps):
            qs[i] = DN_Moments(y[get_window(i)], 3)
    elif windowStat == 'mom4':
        for i in range(numSteps):
            qs[i] = DN_Moments(y[get_window(i)], 4)
    elif windowStat == 'mom5':
        for i in range(numSteps):
            qs[i] = DN_Moments(y[get_window(i)], 5)
    elif windowStat == 'AC1':
        for i in range(numSteps):
            qs[i] = CO_AutoCorr(y[get_window(i)], 1, 'Fourier')
    elif windowStat == 'lillie':
        warnings.warn(f"{windowStat} not yet implemented")
    else:
        raise ValueError(f"Unknown statistic '{windowStat}'")
    

    if acrossWinStat == 'std':
        out = np.std(qs, ddof=1)/np.std(y, ddof=1)
    elif acrossWinStat == 'apen':
        out = EN_ApEN(qs, 1, 0.2)
    elif acrossWinStat == 'sampen':
        warnings.warn(f"{acrossWinStat} not yet implemented")
    elif acrossWinStat == 'ent':
        warnings.warn(f"{acrossWinStat} not yet implemented")
    else:
        raise ValueError(f"Unknown statistic '{acrossWinStat}'")
    
    return out
