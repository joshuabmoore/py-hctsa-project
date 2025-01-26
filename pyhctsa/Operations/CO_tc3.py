import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_FirstMin import CO_FirstMin

def CO_tc3(y, tau = 'ac'):
    """
    Normalized nonlinear autocorrelation function, tc3.

    Computes the tc3 function, a normalized nonlinear autocorrelation, at a
    given time-delay, tau.
    Statistic is for two time-delays, normalized in terms of a single time delay.
    Used as a test statistic for higher order correlational moments in surrogate
    data analysis.

    Parameters:
    y (array-like): Input time series
    tau (int or str, optional): Time lag. If 'ac' or 'mi', it will be computed.

    Returns:
    dict: A dictionary containing:
        - 'raw': The raw tc3 expression
        - 'abs': The magnitude of the raw expression
        - 'num': The numerator
        - 'absnum': The magnitude of the numerator
        - 'denom': The denominator

    Note: This function requires the implementation of CO_FirstCrossing and 
    CO_FirstMin functions, which are not provided in this conversion.
    """

    # Set the time lag as a measure of the time-series correlation length
    # Can set the time lag, tau, to be 'ac' or 'mi'
    if tau == 'ac':
        # tau is first zero crossing of the autocorrelation function
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = CO_FirstMin(y, 'mi')
    
    if np.isnan(tau):
        raise ValueError("No valid setting for time delay (time series too short?)")
    
    # Compute tc3 statistic
    yn = y[:-2*tau]
    yn1 = y[tau:-tau] # yn1, tau steps ahead
    yn2 = y[2*tau:] # yn2, 2*tau steps ahead

    numerator = np.mean(yn * yn1 * yn2)
    denominator = np.abs(np.mean(yn * yn1)) ** (3/2)

    # The expression used in TSTOOL tc3:
    out = {}
    out['raw'] = numerator / denominator

    # The magnitude
    out['abs'] = np.abs(out['raw'])

    # The numerator
    out['num'] = numerator
    out['absnum'] = np.abs(out['num'])

    # The denominator
    out['denom'] = denominator

    return out
