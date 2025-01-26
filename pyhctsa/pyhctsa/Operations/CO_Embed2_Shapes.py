import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_AutoCorr import CO_AutoCorr
from Utils.binpicker import binpicker

def CO_Embed2_Shapes(y, tau = 'tau', shape = 'circle', r = 1):
    """
    Shape-based statistics in a 2-d embedding space.

    Takes a shape and places it on each point in the two-dimensional time-delay
    embedding space sequentially. This function counts the points inside this shape
    as a function of time, and returns statistics on this extracted time series.

    Parameters:
    -----------
    y : array_like
        The input time-series as a (z-scored) column vector.
    tau : int or str, optional
        The time-delay. If 'tau', it's set to the first zero crossing of the autocorrelation function.
    shape : str, optional
        The shape to use. Currently only 'circle' is supported.
    r : float, optional
        The radius of the circle.

    Returns:
    --------
    dict
        A dictionary containing various statistics of the constructed time series.
    """
    if tau == 'tau':
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
        # cannot set time delay > 10% of the length of the time series...
        if tau > len(y)/10:
            tau = int(np.floor(len(y)/10))
        
    # Create the recurrence space, populated by points m
    m = np.column_stack((y[:-tau], y[tau:]))
    N = len(m)

    # Start the analysis
    counts = np.zeros(N)
    if shape == 'circle':
        # Puts a circle around each point in the embedding space in turn
        # counts how many pts are inside this shape, looks at the time series thus formed
        for i in range(N): # across all pts in the time series
            m_c = m - m[i] # pts wrt current pt i
            m_c_d = np.sum(m_c**2, axis=1) # Euclidean distances from pt i
            counts[i] = np.sum(m_c_d <= r**2) # number of pts enclosed in a circle of radius r
    else:
        raise ValueError(f"Unknown shape '{shape}'")
    
    counts -= 1 # ignore self counts

    if np.all(counts == 0):
        print("No counts detected!")
        return np.nan

    # Return basic statistics on the counts
    out = {}
    out['ac1'] = CO_AutoCorr(counts, 1, 'Fourier')[0]
    out['ac2'] = CO_AutoCorr(counts, 2, 'Fourier')[0]
    out['ac3'] = CO_AutoCorr(counts, 3, 'Fourier')[0]
    out['tau'] = CO_FirstCrossing(counts, 'ac', 0, 'continuous')
    out['max'] = np.max(counts)
    out['std'] = np.std(counts, ddof=1)
    out['median'] = np.median(counts)
    out['mean'] = np.mean(counts)
    out['iqr'] = np.percentile(counts, 75, method='hazen') - np.percentile(counts, 25, method='hazen')
    out['iqronrange'] = out['iqr']/np.ptp(counts)

    # distribution - using sqrt binning method
    numBinsToUse = int(np.ceil(np.sqrt(len(counts)))) # supposed to be what MATLAB uses for 'sqrt' option.
    minX, maxX = np.min(counts), np.max(counts)
    binWidthEst = (maxX - minX)/numBinsToUse
    edges = binpicker(minX, maxX, None, binWidthEst) # mimics the functionality of MATLAB's internal function for selecting bins
    binCounts, binEdges = np.histogram(counts, bins=edges)
    # normalise bin counts
    binCountsNorm = np.divide(binCounts, np.sum(binCounts))
    # get bin centres
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    out['mode_val'] = np.max(binCountsNorm)
    out['mode'] = binCentres[np.argmax(binCountsNorm)]
    # histogram entropy
    out['hist_ent'] = np.sum(binCountsNorm[binCountsNorm > 0] * np.log(binCountsNorm[binCountsNorm > 0]))

    # Stationarity measure for fifths of the time series
    afifth = int(np.floor(N/5))
    buffer_m = np.array([counts[i*afifth:(i+1)*afifth] for i in range(5)])
    out['statav5_m'] = np.std(np.mean(buffer_m, axis=1), ddof=1) / np.std(counts, ddof=1)
    out['statav5_s'] = np.std(np.std(buffer_m, axis=1, ddof=1), ddof=1) / np.std(counts, ddof=1)


    return out
