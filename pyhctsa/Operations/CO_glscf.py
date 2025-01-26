import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing

def CO_glscf(y, alpha, beta, tau = 'tau'):
    """
    The generalized linear self-correlation function of a time series.
    
    This function was introduced in Queiros and Moyano in Physica A, Vol. 383, pp.
    10-15 (2007) in the paper "Yet on statistical properties of traded volume:
    Correlation and mutual information at different value magnitudes".
    https://www.sciencedirect.com/science/article/pii/S0378437107004645
    
    The function considers magnitude correlations.
    
    Parameters
    ----------
    y : array_like
        The input time series.
    alpha : float
        A real and nonzero parameter.
    beta : float
        A real and nonzero parameter.
    tau : int or str, optional
        The time-delay. Can also be 'tau' to set to first zero-crossing of the ACF.
    
    Returns
    -------
    glscf : float
        The generalized linear self-correlation function value.
    """
    # Set tau to first zero-crossing of the autocorrelation function with the input 'tau'
    if tau == 'tau':
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Take magnitudes of time-delayed versions of the time series
    y1 = np.abs(y[:-tau])
    y2 = np.abs(y[tau:])


    p1 = np.mean(np.multiply((y1 ** alpha), (y2 ** beta)))
    p2 = np.multiply(np.mean(y1 ** alpha), np.mean(y2 ** beta))
    p3 = np.sqrt(np.mean(y1 ** (2*alpha)) - (np.mean(y1 ** alpha))**2)
    p4 = np.sqrt(np.mean(y2 ** (2*beta)) - (np.mean(y2 ** beta))**2)

    glscf = (p1 - p2) / (p3 * p4)

    return glscf    
