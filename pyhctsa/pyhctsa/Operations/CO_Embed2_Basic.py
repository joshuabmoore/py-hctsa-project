import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing

def CO_Embed2_Basic(y, tau=1):
    """
    Point density statistics in a 2-d embedding space.

    Computes a set of point-density statistics in a plot of y_i against y_{i-tau}.

    Parameters:
    -----------
    y : array_like
        The input time series.
    tau : int or str, optional
        The time lag (can be set to 'tau' to set the time lag to the first zero
        crossing of the autocorrelation function).

    Returns:
    --------
    out : dict
        Dictionary containing various point density statistics.
    """

    if tau == 'tau':
        # Make tau the first zero crossing of the autocorrelation function
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')

    xt = y[:-tau]  # part of the time series
    xtp = y[tau:]  # time-lagged time series
    N = len(y) - tau  # Length of each time series subsegment

    out = {}

    # Points in a thick bottom-left -- top-right diagonal
    out['updiag01'] = np.sum(np.abs(xtp - xt) < 0.1) / N
    out['updiag05'] = np.sum(np.abs(xtp - xt) < 0.5) / N

    # Points in a thick bottom-right -- top-left diagonal
    out['downdiag01'] = np.sum(np.abs(xtp + xt) < 0.1) / N
    out['downdiag05'] = np.sum(np.abs(xtp + xt) < 0.5) / N

    # Ratio of these
    out['ratdiag01'] = out['updiag01'] / out['downdiag01']
    out['ratdiag05'] = out['updiag05'] / out['downdiag05']

    # In a thick parabola concave up
    out['parabup01'] = np.sum(np.abs(xtp - xt**2) < 0.1) / N
    out['parabup05'] = np.sum(np.abs(xtp - xt**2) < 0.5) / N

    # In a thick parabola concave down
    out['parabdown01'] = np.sum(np.abs(xtp + xt**2) < 0.1) / N
    out['parabdown05'] = np.sum(np.abs(xtp + xt**2) < 0.5) / N

    # In a thick parabola concave up, shifted up 1
    out['parabup01_1'] = np.sum(np.abs(xtp - (xt**2 + 1)) < 0.1) / N
    out['parabup05_1'] = np.sum(np.abs(xtp - (xt**2 + 1)) < 0.5) / N

    # In a thick parabola concave down, shifted up 1
    out['parabdown01_1'] = np.sum(np.abs(xtp + (xt**2 - 1)) < 0.1) / N
    out['parabdown05_1'] = np.sum(np.abs(xtp + (xt**2 - 1)) < 0.5) / N

    # In a thick parabola concave up, shifted down 1
    out['parabup01_n1'] = np.sum(np.abs(xtp - (xt**2 - 1)) < 0.1) / N
    out['parabup05_n1'] = np.sum(np.abs(xtp - (xt**2 - 1)) < 0.5) / N

    # In a thick parabola concave down, shifted down 1
    out['parabdown01_n1'] = np.sum(np.abs(xtp + (xt**2 + 1)) < 0.1) / N
    out['parabdown05_n1'] = np.sum(np.abs(xtp + (xt**2 + 1)) < 0.5) / N

    # RINGS (points within a radius range)
    out['ring1_01'] = np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.1) / N
    out['ring1_02'] = np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.2) / N
    out['ring1_05'] = np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.5) / N

    # CIRCLES (points inside a given circular boundary)
    out['incircle_01'] = np.sum(xtp**2 + xt**2 < 0.1) / N
    out['incircle_02'] = np.sum(xtp**2 + xt**2 < 0.2) / N
    out['incircle_05'] = np.sum(xtp**2 + xt**2 < 0.5) / N
    out['incircle_1'] = np.sum(xtp**2 + xt**2 < 1) / N
    out['incircle_2'] = np.sum(xtp**2 + xt**2 < 2) / N
    out['incircle_3'] = np.sum(xtp**2 + xt**2 < 3) / N
    
    incircle_values = [out['incircle_01'], out['incircle_02'], out['incircle_05'],
                       out['incircle_1'], out['incircle_2'], out['incircle_3']]
    out['medianincircle'] = np.median(incircle_values)
    out['stdincircle'] = np.std(incircle_values, ddof=1)
    
    return out
