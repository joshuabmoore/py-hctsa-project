import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing

def CO_HistogramAMI(y, tau = 1, meth = 'even', numBins = 10):
    """
    CO_HistogramAMI: The automutual information of the distribution using histograms.

    Parameters:
    y (array-like): The input time series
    tau (int, list or str): The time-lag(s) (default: 1)
    meth (str): The method of computing automutual information:
                'even': evenly-spaced bins through the range of the time series,
                'std1', 'std2': bins that extend only up to a multiple of the
                                standard deviation from the mean of the time series to exclude outliers,
                'quantiles': equiprobable bins chosen using quantiles.
    num_bins (int): The number of bins (default: 10)

    Returns:
    float or dict: The automutual information calculated in this way.
    """
    # Use first zero crossing of the ACF as the time lag
    if isinstance(tau, str) and tau in ['ac', 'tau']:
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Bins for the data
    # same for both -- assume same distribution (true for stationary processes, or small lags)
    if meth == 'even':
        b = np.linspace(np.min(y), np.max(y), numBins + 1)
        # Add increment buffer to ensure all points are included
        inc = 0.1
        b[0] -= inc
        b[-1] += inc
    elif meth == 'std1': # bins out to +/- 1 std
        b = np.linspace(-1, 1, numBins + 1)
        if np.min(y) < -1:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 1:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'std2': # bins out to +/- 2 std
        b = np.linspace(-2, 2, numBins + 1)
        if np.min(y) < -2:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 2:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'quantiles': # use quantiles with ~equal number in each bin
        b = np.quantile(y, np.linspace(0, 1, numBins + 1), method='hazen')
        b[0] -= 0.1
        b[-1] += 0.1
    else:
        raise ValueError(f"Unknown method '{meth}'")
    
    # Sometimes bins can be added (e.g., with std1 and std2), so need to redefine numBins
    numBins = len(b) - 1

    # Form the time-delay vectors y1 and y2
    if not isinstance(tau, (list, np.ndarray)):
        # if only single time delay as integer, make into a one element list
        tau = [tau]

    amis = np.zeros(len(tau))

    for i, t in enumerate(tau):
        if t == 0:
            # for tau = 0, y1 and y2 are identical to y
            y1 = y2 = y
        else:
            y1 = y[:-t]
            y2 = y[t:]
        # Joint distribution of y1 and y2
        pij, _, _ = np.histogram2d(y1, y2, bins=(b, b))
        pij = pij[:numBins, :numBins]  # joint
        pij = pij / np.sum(pij)  # normalize
        pi = np.sum(pij, axis=1)  # marginal
        pj = np.sum(pij, axis=0)  # other marginal

        pii = np.tile(pi, (numBins, 1)).T
        pjj = np.tile(pj, (numBins, 1))

        r = pij > 0  # Defining the range in this way, we set log(0) = 0
        amis[i] = np.sum(pij[r] * np.log(pij[r] / pii[r] / pjj[r]))

    if len(tau) == 1:
        return amis[0]
    else:
        return {f'ami{i+1}': ami for i, ami in enumerate(amis)}
