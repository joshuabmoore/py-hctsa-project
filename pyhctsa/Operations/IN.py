import numpy as np
import jpype as jp
from loguru import logger
import os
from scipy import stats
from CO import FirstCrossing, AutoCorr, HistogramAMI
from typing import Union
from scipy.optimize import curve_fit
from BF_iszscored import BF_iszscored
from BF_MutualInformation import BF_MutualInformation

def FirstMin(y : list, minWhat : str = 'mi-gaussian', extraParam : any = None, minNotMax : Union[bool, None] = True):
    """
    Time of first minimum in a given self-correlation function.

    Parameters
    ----------
    y : array-like
        The input time series.
    minWhat : str, optional
        The type of correlation to minimize. Options are 'ac' for autocorrelation,
        or 'mi' for automutual information. By default, 'mi' specifies the
        'gaussian' method from the Information Dynamics Toolkit. Other options
        include 'mi-kernel', 'mi-kraskov1', 'mi-kraskov2' (from Information Dynamics Toolkit),
        or 'mi-hist' (histogram-based method). Default is 'mi'.
    extraParam : any, optional
        An additional parameter required for the specified `minWhat` method (e.g., for Kraskov).
    minNotMax : bool, optional
        If True, return the maximum instead of the minimum. Default is False.

    Returns
    -------
    int
        The time of the first minimum (or maximum if `minNotMax` is True).
    """

    N = len(y)

    # Define the autocorrelation function
    if minWhat in ['ac', 'corr']:
        # Autocorrelation implemented as CO_AutoCorr
        corrfn = lambda x : AutoCorr(y, tau=x, method='Fourier')
    elif minWhat == 'mi-hist':
        # if extraParam is none, use default num of bins in BF_MutualInformation (default : 10)
        corrfn = lambda x : BF_MutualInformation(y[:-x], y[x:], 'range', 'range', extraParam or 10)
    elif minWhat == 'mi-kraskov2':
        # (using Information Dynamics Toolkit)
        # extraParam is the number of nearest neighbors
        corrfn = lambda x : AutoMutualInfo(y, x, 'kraskov2', extraParam)
    elif minWhat == 'mi-kraskov1':
        # (using Information Dynamics Toolkit)
        corrfn = lambda x : AutoMutualInfo(y, x, 'kraskov1', extraParam)
    elif minWhat == 'mi-kernel':
        corrfn = lambda x : AutoMutualInfo(y, x, 'kernel', extraParam)
    elif minWhat in ['mi', 'mi-gaussian']:
        corrfn = lambda x : AutoMutualInfo(y, x, 'gaussian', extraParam)
    else:
        raise ValueError(f"Unknown correlation type specified: {minWhat}")
    
    # search for a minimum (incrementally through time lags until a minimum is found)
    autoCorr = np.zeros(N-1) # pre-allocate maximum length autocorrelation vector
    if minNotMax:
        # FIRST LOCAL MINUMUM 
        for i in range(1, N):
            autoCorr[i-1] = corrfn(i)
            # Hit a NaN before got to a minimum -- there is no minimum
            if np.isnan(autoCorr[i-1]):
                logger.warning(f"No minimum in {minWhat} [[time series too short to find it?]]")
                return np.nan
            
            # we're at a local minimum
            if (i == 2) and (autoCorr[1] > autoCorr[0]):
                # already increases at lag of 2 from lag of 1: a minimum (since ac(0) is maximal)
                return 1
            elif (i > 2) and autoCorr[i-3] > autoCorr[i-2] < autoCorr[i-1]:
                # minimum at previous i
                return i-1 # I found the first minimum!
    else:
        # FIRST LOCAL MAXIMUM
        for i in range(1, N):
            autoCorr[i-1] = corrfn(i)
            # Hit a NaN before got to a max -- there is no max
            if np.isnan(autoCorr[i-1]):
                logger.warning(f"No minimum in {minWhat} [[time series too short to find it?]]")
                return np.nan

            # we're at a local maximum
            if i > 2 and autoCorr[i-3] < autoCorr[i-2] > autoCorr[i-1]:
                return i-1

    return np.nan

def AddNoise(y : list, tau : int = 1, amiMethod : str = 'even', extraParam : str = 10, randomSeed : Union[None, int]  = None) -> dict:
    """
    Changes in the automutual information with the addition of noise.

    Parameters:
    y (array-like): The input time series (should be z-scored)
    tau (int or str): The time delay for computing AMI (default: 1)
    amiMethod (str): The method for computing AMI:
                      'std1','std2','quantiles','even' for histogram-based estimation,
                      'gaussian','kernel','kraskov1','kraskov2' for estimation using JIDT
    extraParam: e.g., the number of bins input to CO_HistogramAMI, or parameter for AutoMutualInfo
    randomSeed (int): Settings for resetting the random seed for reproducible results

    Returns:
    dict: Statistics on the resulting set of automutual information estimates
    """

    if not BF_iszscored(y):
        logger.warning("Input time series should be z-scored.")
    
    # Set tau to minimum of autocorrelation function if 'ac' or 'tau'
    if tau in ['ac', 'tau']:
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Generate noise
    if randomSeed is not None:
        np.random.seed(randomSeed)
    else:
        np.random.seed(0)
    noise = np.random.randn(len(y)) # generate uncorrelated additive noise

    # Set up noise range
    noiseRange = np.linspace(0, 3, 50) # compare properties across this noise range
    numRepeats = len(noiseRange)

    # Compute the automutual information across a range of noise levels
    amis = np.zeros(numRepeats)
    if amiMethod in ['std1', 'std2', 'quantiles', 'even']:
        # histogram-based methods using my naive implementation in CO_Histogram
        for i in range(numRepeats):
            amis[i] = HistogramAMI(y + noiseRange[i]*noise, tau, amiMethod, extraParam)
            if np.isnan(amis[i]):
                raise ValueError('Error computing AMI: Time series too short (?)')
    if amiMethod in ['gaussian','kernel','kraskov1','kraskov2']:
        for i in range(numRepeats):
            amis[i] = AutoMutualInfo(y + noiseRange[i]*noise, tau, amiMethod, extraParam)
            if np.isnan(amis[i]):
                raise ValueError('Error computing AMI: Time series too short (?)')
    
    # Output statistics
    out = {}
    # Proportion decreases
    out['pdec'] = np.sum(np.diff(amis) < 0) / (numRepeats - 1)

    # Mean change in AMI
    out['meanch'] = np.mean(np.diff(amis))

    # Autocorrelation of AMIs
    out['ac1'] = AutoCorr(amis, 1, 'Fourier')
    out['ac2'] = AutoCorr(amis, 2, 'Fourier')

    # Noise level required to reduce ami to proportion x of its initial value
    firstUnderVals = [0.75, 0.50, 0.25]
    for val in firstUnderVals:
        out[f'firstUnder{val*100}'] = _firstUnder_fn(val * amis[0], noiseRange, amis)

    # AMI at actual noise levels: 0.5, 1, 1.5 and 2
    noiseLevels = [0.5, 1, 1.5, 2]
    for nlvl in noiseLevels:
        out[f'ami_at_{nlvl*10}'] = amis[np.argmax(noiseRange >= nlvl)]

    # Count number of times the AMI function crosses its mean
    out['pcrossmean'] = np.sum(np.diff(np.sign(amis - np.mean(amis))) != 0) / (numRepeats - 1)

    # Fit exponential decay model 
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

def _firstUnder_fn(x, m, p):
    """
    Find the value of m for the first time p goes under the threshold, x. 
    p and m are vectors of the same length
    """
    return next((m_val for m_val, p_val in zip(m, p) if p_val < x), m[-1])

def AutoMutualInfoStats(y : list, maxTau : any = None, estMethod : str = 'kernel', extraParam : any = None):
    """
    Statistics on automutual information function of a time series.

    Parameters:
    ----------
    y (array-like) : column vector of time series.
    estMethod (str) : input to AutoMutualInfo
    extraParam (str, int, optional) : input to AutoMutualInfo
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
    ami = AutoMutualInfo(y, timeDelay=timeDelay, estMethod=estMethod, extraParam=extraParam)
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
    out['amiac1'] = AutoCorr(ami, 1, 'Fourier')[0]

    return out 


def _initialize_MI(estMethod : str, extraParam : any = None, addNoise : bool = False):
    """
    Helper function to initialize mutual information calculator.
    """

    # Check to see whether a jpype JVM has been started.
    if not jp.isJVMStarted():
        jarloc = (
            os.path.dirname(os.path.abspath(__file__)) + "/../Toolboxes/infodynamics-dist/infodynamics.jar"
        )
        # change to debug info
        logger.debug(f"Starting JVM with java class {jarloc}.")
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)


    if estMethod == 'gaussian':
        implementingClass = 'infodynamics.measures.continuous.gaussian'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateGaussian()
    elif estMethod == 'kernel':
        implementingClass = 'infodynamics.measures.continuous.kernel'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKernel()
    elif estMethod == 'kraskov1':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov1()
    elif estMethod == 'kraskov2':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov2()
    else:
        raise ValueError(f"Unknown mutual information estimation method '{estMethod}'")

    # Add neighest neighbor option for KSG estimator
    if estMethod in ['kraskov1', 'kraskov2']:
        if extraParam != None:
            if isinstance(extraParam, int):
                logger.warning("Number of nearest neighbors needs to be a string. Setting this for you...")
                extraParam = str(extraParam)
            miCalc.setProperty('k', extraParam) # 4th input specifies number of nearest neighbors for KSG estimator
        else:
            miCalc.setProperty('k', '3') # use 3 nearest neighbors for KSG estimator as default
        
    # Make deterministic if kraskov1 or 2 (which adds a small amount of noise to the signal by default)
    if (estMethod in ['kraskov1', 'kraskov2']) and (addNoise == False):
        miCalc.setProperty('NOISE_LEVEL_TO_ADD','0')
    
    # Specify a univariate calculation
    miCalc.initialise(1,1)

    return miCalc

def AutoMutualInfo(y : list, timeDelay : int = 1, estMethod : str = 'kernel', extraParam : any = None):
    """
    Time-series automutual information

    Parameters:
    -----------
    y : array_like
        Input time series (column vector)
    time_delay : int or list, optional
        Time lag for automutual information calculation (default is 1)
    estMethod : str, optional
        The estimation method used to compute the mutual information:
        - 'gaussian'
        - 'kernel' (default)
        - 'kraskov1'
        - 'kraskov2'
    extraParam : any, optional
        Extra parameters for the estimation method (default is None)

    Returns:
    --------
    out : float or dict
        Automutual information value(s)
    """

    if isinstance(timeDelay, str) and timeDelay in ['ac', 'tau']:
        timeDelay = FirstCrossing(y, corr_fun='ac', threshold=0, what_out='discrete')
        
    y = np.asarray(y).flatten()
    N = len(y)
    minSamples = 5 # minimum 5 samples to compute mutual information (could make higher?)

    # Loop over time delays if a vector
    if not isinstance(timeDelay, list):
        timeDelay = [timeDelay]
    
    numTimeDelays = len(timeDelay)
    amis = np.full(numTimeDelays, np.nan)

    if numTimeDelays > 1:
        timeDelay = np.sort(timeDelay)
    
    # initialise the MI calculator object if using non-Gaussian estimator
    if estMethod != 'gaussian':
        # assumes the JVM has already been started up
        miCalc = _initialize_MI(estMethod, extraParam=extraParam, addNoise=False) # NO ADDED NOISE!
    
    for k, delay in enumerate(timeDelay):
        # check enough samples to compute automutual info
        if delay > N - minSamples:
            # time sereis too short - keep the remaining values as NaNs
            break
        # form the time-delay vectors y1 and y2
        y1 = y[:-delay]
        y2 = y[delay:]

        if estMethod == 'gaussian':
            r, _ = stats.pearsonr(y1, y2)
            amis[k] = -0.5*np.log(1 - r**2)
        else:
            # Reinitialize for Kraskov:
            miCalc.initialise(1, 1)
            # Set observations to time-delayed versions of the time series:
            y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
            y2_jp = jp.JArray(jp.JDouble)(y2)
            miCalc.setObservations(y1_jp, y2_jp)
            # compute
            amis[k] = miCalc.computeAverageLocalOfObservations()
        
    if np.isnan(amis).any():
        print(f"Warning: Time series (N={N}) is too short for automutual information calculations up to lags of {max(timeDelay)}")
    if numTimeDelays == 1:
        # return a scalar if only one time delay
        return amis[0]
    else:
        # return a dict for multiple time delays
        return {f"ami{delay}": ami for delay, ami in zip(timeDelay, amis)}

def MutualInfo(y1, y2, estMethod = 'kernel', extraParam = None):
    """
    Compute the mutual information of two data vectors using the information dynamics toolkit implementation.

    Parameters:
    -----------
    y1 : array-like
        Input time series 1.
    y2 : array-like
        Input time series 2.
    estMethod : str
        The estimation method used to compute the mutual information. Options are:
        - 'gaussian'
        - 'kernel'
        - 'kraskov1'
        - 'kraskov2'

        Refer to:
        Kraskov, A., Stoegbauer, H., Grassberger, P. (2004). Estimating mutual information.
        Physical Review E, 69(6), 066138. DOI: http://dx.doi.org/10.1103/PhysRevE.69.066138

    Returns:
    --------
    float
        The estimated mutual information between the two input time series.
    """
    # Initialize miCalc object (don't add noise!):
    miCalc = _initialize_MI(estMethod=estMethod, extraParam=extraParam, addNoise=False)
    # Set observations to two time series:
    y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
    y2_jp = jp.JArray(jp.JDouble)(y2) # convert observations to java double
    miCalc.setObservations(y1_jp, y2_jp)

    # Compute mutual information
    out = miCalc.computeAverageLocalOfObservations()

    return out

