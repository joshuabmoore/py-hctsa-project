import numpy as np
import jpype as jp
from scipy import stats
from Operations.IN_Initialize_MI import IN_Initialize_MI
from Operations.CO_FirstCrossing import CO_FirstCrossing

def IN_AutoMutualInfo(y, timeDelay=1, estMethod='kernel', extraParam=None):
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
        timeDelay = CO_FirstCrossing(y, corr_fun='ac', threshold=0, what_out='discrete')
        
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
        miCalc = IN_Initialize_MI(estMethod, extraParam=extraParam, addNoise=False) # NO ADDED NOISE!
    
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
