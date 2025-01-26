from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_FirstMin import CO_FirstMin
import numpy as np

def CO_trev(y, tau = 'ac'):
    """
    Normalized nonlinear autocorrelation, trev function of a time series.

    Calculates the trev function, a normalized nonlinear autocorrelation,
    mentioned in the documentation of the TSTOOL nonlinear time-series analysis
    package.

    Parameters:
    y (array-like): Time series
    tau (int, str, optional): Time lag. Can be 'ac' or 'mi' to set as the first 
                              zero-crossing of the autocorrelation function, or 
                              the first minimum of the automutual information 
                              function, respectively. Default is 'ac'.

    Returns:
    dict: A dictionary containing the following keys:
        - 'raw': The raw trev expression
        - 'abs': The magnitude of the raw expression
        - 'num': The numerator
        - 'absnum': The magnitude of the numerator
        - 'denom': The denominator

    Raises:
    ValueError: If no valid setting for time delay is found.
    """

    # Can set the time lag, tau, to be 'ac' or 'mi'
    if tau == 'ac':
        # tau is first zero crossing of the autocorrelation function
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = CO_FirstMin(y, 'mi')
    if np.isnan(tau):
        raise ValueError("No valid setting for time delay. (Is the time series too short?)")

    # Compute trev quantities
    yn = y[:-tau]
    yn1 = y[tau:] # yn, tau steps ahead
    
    out = {}

    # The trev expression used in TSTOOL
    raw = np.mean((yn1 - yn)**3) / (np.mean((yn1 - yn)**2))**(3/2)
    out['raw'] = raw

    # The magnitude
    out['abs'] = np.abs(raw)

    # The numerator
    num = np.mean((yn1-yn)**3)
    out['num'] = num
    out['absnum'] = np.abs(num)

    # the denominator
    out['denom'] = (np.mean((yn1-yn)**2))**(3/2)

    return out
