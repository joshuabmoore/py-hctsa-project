import numpy as np
from statsmodels.tsa.stattools import pacf

def CO_PartialAutoCorr(y, max_tau=10, what_method='ols'):
    """
    Compute the partial autocorrelation of an input time series.

    Parameters:
    ----------
    y (array-like): A scalar time series column vector.
    max_tau (int): The maximum time-delay. Returns for lags up to this maximum.
    what_method (str): The method used to compute: 'ols' or 'yw' (Yule-Walker).

    Returns:
    ----------
    out (dict): The partial autocorrelations across the set of time lags.

    Raises:
    ----------
    ValueError: If max_tau is negative or what_method is invalid.
    """
    y = np.array(y)
    N = len(y)  # time-series length

    if max_tau <= 0:
        raise ValueError('Negative or zero time lags not applicable')

    method_map = {'ols': 'ols', 'yule_walker': 'yw'}
    if what_method not in method_map:
        raise ValueError(f"Invalid method: {what_method}. Use 'ols' or 'yule_walker'.")

    # Compute partial autocorrelation
    pacf_values = pacf(y, nlags=max_tau, method=method_map[what_method])

    # Create output dictionary
    out = {}
    for i in range(1, max_tau + 1):
        out[f'pac_{i}'] = pacf_values[i]

    return out
