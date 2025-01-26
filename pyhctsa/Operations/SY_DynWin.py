import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_AutoCorr import CO_AutoCorr
from Operations.EN_SampEn import EN_SampEn
from scipy.stats import skew, kurtosis

def SY_DynWin(y, maxNumSegments = 10):
    """
    How stationarity estimates depend on the number of time-series subsegments
    
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
    taug = CO_FirstCrossing(y, 'ac', 0, 'discrete') # global tau

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
            taul = CO_FirstCrossing(ySub, 'ac', 0, 'discrete')

            qs[j, 0] = np.mean(ySub)
            qs[j, 1] = np.std(ySub, ddof=1)
            qs[j, 2] = skew(ySub)
            qs[j, 3] = kurtosis(ySub)
            sampenOut = EN_SampEn(ySub, 2, 0.15)
            qs[j, 4] = sampenOut['quadSampEn1'] # SampEn_1_015
            qs[j, 5] = sampenOut['quadSampEn2'] # SampEn_2_015
            qs[j, 6] = CO_AutoCorr(ySub, 1, 'Fourier')[0] # AC1
            qs[j, 7] = CO_AutoCorr(ySub, 2, 'Fourier')[0] # AC2
            # (Sometimes taug or taul can be longer than ySub; then these will output NaNs:)
            qs[j, 8] = CO_AutoCorr(ySub, taug, 'Fourier')[0] # AC_glob_tau
            qs[j, 9] = CO_AutoCorr(ySub, taul, 'Fourier')[0] # AC_loc_tau
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
