import numpy as np
from Operations.CO_AutoCorr import CO_AutoCorr

def CO_Embed2_AngleTau(y, maxTau):
    """
    Angle autocorrelation in a 2-dimensional embedding space

    Investigates how the autocorrelation of angles between successive points in
    the two-dimensional time-series embedding change as tau varies from
    tau = 1, 2, ..., maxTau.

    Parameters:
    -----------
    y (numpy.ndarray): Input time series (1D array)
    maxTau (int): The maximum time lag to consider

    Returns:
    --------
    dict: A dictionary containing various statistics
    """
    tauRange = np.arange(1, maxTau + 1)
    numTau = len(tauRange)

    # Ensure y is a column vector
    y = np.atleast_2d(y)
    if y.shape[0] < y.shape[1]:
        y = y.T

    stats_store = np.zeros((3, numTau))

    for i, tau in enumerate(tauRange):
        m = np.column_stack((y[:-tau], y[tau:]))
        theta = np.diff(m[:, 1]) / np.diff(m[:, 0])
        theta = np.arctan(theta)  # measured as deviation from the horizontal

        if len(theta) == 0:
            raise ValueError(f'Time series (N={len(y)}) too short for embedding')

        stats_store[0, i] = CO_AutoCorr(theta, 1, 'Fourier')[0]
        stats_store[1, i] = CO_AutoCorr(theta, 2, 'Fourier')[0]
        stats_store[2, i] = CO_AutoCorr(theta, 3, 'Fourier')[0]
    
    # Compute output statistics
    out = {
        'ac1_thetaac1': CO_AutoCorr(stats_store[0, :], 1, 'Fourier'),
        'ac1_thetaac2': CO_AutoCorr(stats_store[1, :], 1, 'Fourier'),
        'ac1_thetaac3': CO_AutoCorr(stats_store[2, :], 1, 'Fourier'),
        'mean_thetaac1': np.mean(stats_store[0, :]),
        'max_thetaac1': np.max(stats_store[0, :]),
        'min_thetaac1': np.min(stats_store[0, :]),
        'mean_thetaac2': np.mean(stats_store[1, :]),
        'max_thetaac2': np.max(stats_store[1, :]),
        'min_thetaac2': np.min(stats_store[1, :]),
        'mean_thetaac3': np.mean(stats_store[2, :]),
        'max_thetaac3': np.max(stats_store[2, :]),
        'min_thetaac3': np.min(stats_store[2, :]),
    }

    out['meanrat_thetaac12'] = out['mean_thetaac1'] / out['mean_thetaac2']
    out['diff_thetaac12'] = np.sum(np.abs(stats_store[1, :] - stats_store[0, :]))

    return out
