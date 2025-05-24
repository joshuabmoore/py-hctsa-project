import numpy as np

def DN_Burstiness(y):
    """
    Calculate the burstiness statistic of a time series.

    This function returns the 'burstiness' statistic as defined in
    Goh and Barabasi's paper, "Burstiness and memory in complex systems,"
    Europhys. Lett. 81, 48002 (2008).

    Parameters
    ----------
    y : array-like
        The input time series.
    
    Returns
    -------
    dict
        The original burstiness statistic, B, and the improved
        burstiness statistic, B_Kim.
    """
    
    mean = np.mean(y)
    std = np.std(y, ddof=1)

    r = np.divide(std,mean) # coefficient of variation
    B = np.divide((r - 1), (r + 1)) # Original Goh and Barabasi burstiness statistic, B

    # improved burstiness statistic, accounting for scaling for finite time series
    # Kim and Jo, 2016, http://arxiv.org/pdf/1604.01125v1.pdf
    N = len(y)
    p1 = np.sqrt(N+1)*r - np.sqrt(N-1)
    p2 = (np.sqrt(N+1)-2)*r + np.sqrt(N-1)

    B_Kim = np.divide(p1, p2)

    out = {'B': B, 'B_Kim': B_Kim}

    return out
