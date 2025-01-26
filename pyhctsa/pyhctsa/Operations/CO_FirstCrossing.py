from Operations.CO_AutoCorr import CO_AutoCorr
from PeripheryFunctions.BF_PointOfCrossing import BF_PointOfCrossing

def CO_FirstCrossing(y, corr_fun='ac', threshold=0, what_out='both'):
    """
    The first crossing of a given autocorrelation across a given threshold.

    Parameters:
    -----------
    y : array_like
        The input time series
    corr_fun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]
    what_out : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'

    Returns:
    --------
    out : dict or float
        The first crossing information, format depends on what_out
    """
    # Select the self-correlation function
    if corr_fun == 'ac':
        # Autocorrelation at all time lags
        corrs = CO_AutoCorr(y, [], 'Fourier')
    else:
        raise ValueError(f"Unknown correlation function '{corr_fun}'")

    # Calculate point of crossing
    first_crossing_index, point_of_crossing_index = BF_PointOfCrossing(corrs, threshold)

    # Assemble the appropriate output (dictionary or float)
    # Convert from index space (1,2,…) to lag space (0,1,2,…)
    if what_out == 'both':
        out = {
            'firstCrossing': first_crossing_index - 1,
            'pointOfCrossing': point_of_crossing_index - 1
        }
    elif what_out == 'discrete':
        out = first_crossing_index - 1
    elif what_out == 'continuous':
        out = point_of_crossing_index - 1
    else:
        raise ValueError(f"Unknown output format '{what_out}'")

    return out
