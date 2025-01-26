import numpy as np
from Operations.CO_glscf import CO_glscf

def CO_fzcglscf(y, alpha, beta, maxtau = None):
    """
    The first zero-crossing of the generalized self-correlation function.
    
    Returns the first zero-crossing of the generalized self-correlation function
    introduced in Duarte Queiros and Moyano in Physica A, Vol. 383, pp. 10-15
    (2007) in the paper "Yet on statistical properties of traded volume:
    Correlation and mutual information at different value magnitudes".
    Uses CO_glscf to calculate the generalized self-correlations.
    Keeps calculating until the function finds a minimum, and returns this lag.
    
    Parameters
    ----------
    y : array_like
        The input time series.
    alpha : float
        The parameter alpha.
    beta : float
        The parameter beta.
    maxtau : int, optional
        A maximum time delay to search up to (default is the time-series length).
    
    Returns
    -------
    out : float
        The first zero-crossing lag of the generalized self-correlation function.
    """
    N = len(y) # the length of the time series

    if maxtau is None:
        maxtau = N
    
    glscfs = np.zeros(maxtau)

    for i in range(1, maxtau+1):
        tau = i

        glscfs[i-1] = CO_glscf(y, alpha, beta, tau)
        if (i > 1) and (glscfs[i-1]*glscfs[i-2] < 0):
            # Draw a straight line between these two and look at where it hits zero
            out = i - 1 + glscfs[i-1]/(glscfs[i-1]-glscfs[i-2])
            return out
    
    return maxtau # if the function hasn't exited yet, set output to maxtau 
