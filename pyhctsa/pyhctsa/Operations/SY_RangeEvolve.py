import numpy as np

def SY_RangeEvolve(y):
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
