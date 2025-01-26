import numpy as np
import pywt

def CP_wavelet_varchg(y, wName = 'db3', level = 3, maxnchpts = 5, minDelay = 0.01):
    """
    Variance change points in a time series.

    Finds variance change points using PyWavelets, estimating the change points in the time series.

    Parameters:
    -----------
    y : array_like
        The input time series.
    w_name : str, optional
        The name of the mother wavelet to analyze the data with (default: 'db3').
    level : int, optional
        The level of wavelet decomposition (default: 3).
    max_nchpts : int, optional
        The maximum number of change points (default: 5).
    min_delay : float, optional
        The minimum delay between consecutive change points, specified as a proportion
        of the time-series length (default: 0.01).

    Returns:
    --------
    int
        The optimal number of change points.
    """
    N = len(y) # length of the time-series

    if level == 'max':
        level = pywt.dwt_max_level(N, wName)
    
    if 0 < minDelay < 1:
        minDelay = int(np.ceil(minDelay*N))
    
    if pywt.dwt_max_level(N, wName) < level:
        raise ValueError(f"Chosen level, {level}, is too large for this wavelet on this signal.")
    
    # 1. Recover a noisy signal by suppressing an approximation

    coeffs = pywt.wavedec(y, wName, level=level) # returns [cAn, cDn, cDn-1, ..., cD2, cD1]
    # reconstruct detail at the same level
    # to get the highest decomp. level we take the second element, cDn
    det = pywt.upcoef('d', coeffs=coeffs[1], wavelet=wName, level=level, take=N) # details reconstruction at level of decomp. 

    # 2. Replace 2% of the greatest (absolute) values by the mean
    x = np.sort(np.abs(det))
    v2p100 = x[int(np.floor(len(x) * 0.98))]
    det[np.abs(det) > v2p100] = np.mean(det)

    # 3. Use wvarchg to estimate the change points
    
