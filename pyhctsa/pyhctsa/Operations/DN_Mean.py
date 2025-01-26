import numpy as np
from scipy import stats

def DN_Mean(y, mean_type='arithmetic'):
    """
    A given measure of location of a data vector.

    Parameters:
    y (array-like): The input data vector
    mean_type (str): The type of mean to calculate
        'norm' or 'arithmetic': arithmetic mean
        'median': median
        'geom': geometric mean
        'harm': harmonic mean
        'rms': root-mean-square
        'iqm': interquartile mean
        'midhinge': midhinge

    Returns:
    out (float): The calculated mean value

    Raises:
    ValueError: If an unknown mean type is specified

    Notes:
    Harmonic mean only defined for positive values.
    """
    y = np.array(y)
    N = len(y)

    if mean_type in ['norm', 'arithmetic']:
        out = np.mean(y)
    elif mean_type == 'median': # median
        out = np.median(y)
    elif mean_type == 'geom': # geometric mean
        out = stats.gmean(y)
    elif mean_type == 'harm': # harmonic mean
        out = stats.hmean(y)
    elif mean_type == 'rms':
        out = np.sqrt(np.mean(y**2))
    elif mean_type == 'iqm': # interquartile mean, cf. DN_TrimmedMean
        p = np.percentile(y, [25, 75])
        out = np.mean(y[(y >= p[0]) & (y <= p[1])])
    elif mean_type == 'midhinge':  # average of 1st and third quartiles
        p = np.percentile(y, [25, 75])
        out = np.mean(p)
    else:
        raise ValueError(f"Unknown mean type '{mean_type}'")

    return out
