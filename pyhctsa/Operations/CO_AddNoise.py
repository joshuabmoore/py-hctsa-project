import numpy as np
from Operations.CO_HistogramAMI import CO_HistogramAMI
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.IN_AutoMutualInfo import IN_AutoMutualInfo
from Operations.CO_AutoCorr import CO_AutoCorr
from PeripheryFunctions.BF_SignChange import BF_SignChange
from PeripheryFunctions.BF_iszscored import BF_iszscored
from scipy.optimize import curve_fit
import warnings

def CO_AddNoise(y, tau = 1, amiMethod = 'even', extraParam = None, randomSeed = None):
    """
    CO_AddNoise: Changes in the automutual information with the addition of noise

    Parameters:
    y (array-like): The input time series (should be z-scored)
    tau (int or str): The time delay for computing AMI (default: 1)
    amiMethod (str): The method for computing AMI:
                      'std1','std2','quantiles','even' for histogram-based estimation,
                      'gaussian','kernel','kraskov1','kraskov2' for estimation using JIDT
    extraParam: e.g., the number of bins input to CO_HistogramAMI, or parameter for IN_AutoMutualInfo
    randomSeed (int): Settings for resetting the random seed for reproducible results

    Returns:
    dict: Statistics on the resulting set of automutual information estimates
    """

    if not BF_iszscored(y):
        warnings.warn("Input time series should be z-scored")
    
    # Set tau to minimum of autocorrelation function if 'ac' or 'tau'
    if tau in ['ac', 'tau']:
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Generate noise
    if randomSeed is not None:
        np.random.seed(randomSeed)
    noise = np.random.randn(len(y)) # generate uncorrelated additive noise

    # Set up noise range
    noiseRange = np.linspace(0, 3, 50) # compare properties across this noise range
    numRepeats = len(noiseRange)

    # Compute the automutual information across a range of noise levels
    amis = np.zeros(numRepeats)
    if amiMethod in ['std1', 'std2', 'quantiles', 'even']:
        # histogram-based methods using my naive implementation in CO_Histogram
        for i in range(numRepeats):
            amis[i] = CO_HistogramAMI(y + noiseRange[i]*noise, tau, amiMethod, extraParam)
            if np.isnan(amis[i]):
                raise ValueError('Error computing AMI: Time series too short (?)')
    if amiMethod in ['gaussian','kernel','kraskov1','kraskov2']:
        for i in range(numRepeats):
            amis[i] = IN_AutoMutualInfo(y + noiseRange[i]*noise, tau, amiMethod, extraParam)
            if np.isnan(amis[i]):
                raise ValueError('Error computing AMI: Time series too short (?)')
    
    # Output statistics
    out = {}
    # Proportion decreases
    out['pdec'] = np.sum(np.diff(amis) < 0) / (numRepeats - 1)

    # Mean change in AMI
    out['meanch'] = np.mean(np.diff(amis))

    # Autocorrelation of AMIs
    out['ac1'] = CO_AutoCorr(amis, 1, 'Fourier')
    out['ac2'] = CO_AutoCorr(amis, 2, 'Fourier')

    # Noise level required to reduce ami to proportion x of its initial value
    firstUnderVals = [0.75, 0.50, 0.25]
    for val in firstUnderVals:
        out[f'firstUnder{val*100}'] = firstUnder_fn(val * amis[0], noiseRange, amis)

    # AMI at actual noise levels: 0.5, 1, 1.5 and 2
    noiseLevels = [0.5, 1, 1.5, 2]
    for nlvl in noiseLevels:
        out[f'ami_at_{nlvl*10}'] = amis[np.argmax(noiseRange >= nlvl)]

    # Count number of times the AMI function crosses its mean
    out['pcrossmean'] = np.sum(np.diff(np.sign(amis - np.mean(amis))) != 0) / (numRepeats - 1)

    # Fit exponential decay
    expFunc = lambda x, a, b : a * np.exp(b * x)
    popt, pcov = curve_fit(expFunc, noiseRange, amis, p0=[amis[0], -1])
    out['fitexpa'], out['fitexpb'] = popt
    residuals = amis - expFunc(noiseRange, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((amis - np.mean(amis))**2)
    out['fitexpr2'] = 1 - (ss_res / ss_tot)
    out['fitexpadjr2'] = 1 - (1-out['fitexpr2'])*(len(amis)-1)/(len(amis)-2-1)
    out['fitexprmse'] = np.sqrt(np.mean(residuals**2))

    # Fit linear function
    p = np.polyfit(noiseRange, amis, 1)
    out['fitlina'], out['fitlinb'] = p
    lin_fit = np.polyval(p, noiseRange)
    out['linfit_mse'] = np.mean((lin_fit - amis)**2)

    return out

# helper functions
def firstUnder_fn(x, m, p):
    """
    Find the value of m for the first time p goes under the threshold, x. 
    p and m vectors of the same length
    """
    first_i = next((m_val for m_val, p_val in zip(m, p) if p_val < x), m[-1])
    return first_i
