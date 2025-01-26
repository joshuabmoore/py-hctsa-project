import numpy as np

def DN_CustomSkewness(y, whatSkew = 'pearson'):
    """
    Custom skewness measures
    Compute the Pearson or Bowley skewness.

    Parameters:
    -----------
    y : array_like
        Input time series
    whatSkew : str, optional
        The skewness measure to calculate:
            - 'pearson'
            - 'bowley'

    Returns:
    --------
    out : float
        The custom skewness measure.
    """

    if whatSkew == 'pearson':
        out = ((3 * np.mean(y) - np.median(y)) / np.std(y, ddof=1))
    elif whatSkew == 'bowley':
        qs = np.quantile(y, [0.25, 0.5, 0.75], method='hazen')
        out = (qs[2]+qs[0] - 2 * qs[1]) / (qs[2] - qs[0]) 
    
    return out
