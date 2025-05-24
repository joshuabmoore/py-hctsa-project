import numpy as np
import pywt
from wrcoef import wavedec, wrcoef, detcoef

def findMyThreshold(x, det_s, N):
    pr = np.argwhere(det_s < x*np.max(det_s))[0] / N
    if len(pr) == 0:
        return np.nan
    else:
        return pr

def WL_coeffs(y : list, wname : str = 'db3', level : int = 3):
    N = len(y)
    if level == 'max':
        level = pywt.dwt_max_level(N, wname)
        if level == 0:
            raise ValueError("Cannot compute wavelet coefficients (short time series)")
    
    if pywt.dwt_max_level(N, wname) < level:
        raise ValueError(f"Chosen level, {level}, is too large for this wavelet on this signal.")
    
    #coeffs = pywt.wavedec(y, wname, level=level)
    C, L = wavedec(y, wavelet=wname, level=level)
    det = wrcoef(C, L, wname, level)
    det_s = np.sort(np.abs(det))[::-1]

    #%% Return statistics
    out = {}
    out['mean_coeff'] = np.mean(det_s)
    out['max_coeff'] = np.max(det_s)
    out['med_coeff'] = np.median(det_s)

    #% Decay rate stats ('where below _ maximum' = 'wb_m')
    out['wb99m'] = findMyThreshold(0.99, det_s, N)
    out['wb90m'] = findMyThreshold(0.90, det_s, N)
    out['wb75m'] = findMyThreshold(0.75, det_s, N)
    out['wb50m'] = findMyThreshold(0.50, det_s, N)
    out['wb25m'] = findMyThreshold(0.25, det_s, N)
    out['wb10m'] = findMyThreshold(0.10, det_s, N)
    out['wb1m'] = findMyThreshold(0.01, det_s, N)

    return out
