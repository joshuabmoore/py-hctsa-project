import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_AutoCorr import CO_AutoCorr

def CO_Embed2(y, tau = 'tau'):
    """
    Statistics of the time series in a 2-dimensional embedding space

    Embeds the (z-scored) time series in a two-dimensional time-delay
    embedding space with a given time-delay, tau, and outputs a set of
    statistics about the structure in this space, including angular
    distribution, etc.

    Parameters:
    y (array-like): The input time series (will be converted to a column vector)
    tau (int or str, optional): The time-delay. If 'tau', it will be set to the first zero-crossing of ACF.

    Returns:
    dict: A dictionary containing various statistics about the embedded time series
    """

    # Set tau to the first zero-crossing of the autocorrelation function with the 'tau' input
    if tau == 'tau':
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
        if tau > len(y) / 10:
            tau = len(y) // 10
    # Ensure that y is a column vector
    y = np.array(y).reshape(-1, 1)

    # Construct the two-dimensional recurrence space
    m = np.hstack((y[:-tau], y[tau:]))
    N = m.shape[0] # number of points in the recurrence space
    

    # 1) Distribution of angles time series; angles between successive points in this space
    theta = np.divide(np.diff(m[:, 1]), np.diff(m[:, 0]))
    theta = np.arctan(theta) # measured as deviation from the horizontal

    out = {}

    out['theta_ac1'] = CO_AutoCorr(theta, 1, 'Fourier')[0]
    out['theta_ac2'] = CO_AutoCorr(theta, 2, 'Fourier')[0]
    out['theta_ac3'] = CO_AutoCorr(theta, 3, 'Fourier')[0]

    out['theta_mean'] = np.mean(theta)
    out['theta_std'] = np.std(theta, ddof=1)
    
    binEdges = np.linspace(-np.pi/2, np.pi/2, 11) # 10 bins in the histogram
    px, _ = histcounts(theta, binEdges=binEdges, normalization='probability')
    binWidths = np.diff(binEdges)
    out['hist10std'] = np.std(px, ddof=1)
    out['histent'] = -np.sum(px[px>0] * np.log(px[px>0] / binWidths[px>0]))
    

    # Stationarity in fifths of the time series
    # Use histograms with 4 bins
    x = np.linspace(-np.pi/2, np.pi/2, 5) # 4 bins
    afifth = (N-1) // 5 # -1 because angles are correlations *between* points
    n = np.zeros((len(x)-1, 5))
    for i in range(5):
        n[:, i], _ = np.histogram(theta[afifth*i:afifth*(i+1)], bins=x)
        
    n = n / afifth
    
    for i in range(4):
        out[f'stdb{i+1}'] = np.std(n[:, i], ddof=1)

    # STATIONARITY of points in the space (do they move around in the space)
    # (1) in terms of distance from origin
    afifth = N // 5
    buffer_m = [m[afifth*i:afifth*(i+1), :] for i in range(5)]

    # Mean euclidean distance in each segment
    eucdm = [np.mean(np.sqrt(x[:, 0]**2 + x[:, 1]**2)) for x in buffer_m]
    for i in range(5):
        out[f'eucdm{i+1}'] = eucdm[i]
    out['std_eucdm'] = np.std(eucdm, ddof=1)
    out['mean_eucdm'] = np.mean(eucdm)

    # Standard deviation of Euclidean distances in each segment
    eucds = [np.std(np.sqrt(x[:, 0]**2 + x[:, 1]**2), ddof=1) for x in buffer_m]
    for i in range(5):
        out[f'eucds{i+1}'] = eucds[i]
    out['std_eucds'] = np.std(eucds, ddof=1)
    out['mean_eucds'] = np.mean(eucds)

    # Maximum volume in each segment (defined as area of rectangle of max span in each direction)
    maxspanx = [np.ptp(x[:, 0]) for x in buffer_m]
    maxspany = [np.ptp(x[:, 1]) for x in buffer_m]
    spanareas = np.multiply(maxspanx, maxspany)
    out['stdspana'] = np.std(spanareas, ddof=1)
    out['meanspana'] = np.mean(spanareas)

    # Outliers in the embedding space
    # area of max span of all points; versus area of max span of 50% of points closest to origin
    d = np.sqrt(m[:, 0]**2 + m[:, 1]**2)
    ix = np.argsort(d)
    
    out['areas_all'] = np.ptp(m[:, 0]) * np.ptp(m[:, 1])
    r50 = ix[:int(np.ceil(len(ix)/2))] # ceil to match MATLAB's round fn output
    
    out['areas_50'] = np.ptp(m[r50, 0]) * np.ptp(m[r50, 1])
    out['arearat'] = out['areas_50'] / out['areas_all']

    return out 

# helper function
def histcounts(x, bins=None, binEdges=None, normalization='probability'):
    """
    Compute histogram bin counts with optional normalization.

    Parameters:
    - x: Input data
    - bins: Number of bins or 'auto' (default)
    - binEdges: Specific bin edges
    - normalization: Normalization method ('count', 'countdensity', 'cumcount',
                     'probability', 'percentage', 'pdf', 'cdf')

    Returns:
    - n: Array of histogram bin counts (normalized if specified)
    - edges: Array of bin edges
    """
    x = np.asarray(x).flatten()
    if binEdges is not None:
        edges = np.asarray(binEdges)
    elif bins is None or bins == 'auto':
        # Use Scott's rule for automatic binning
        bin_width = 3.5 * np.std(x, ddof=1) / (len(x) ** (1/3))
        edges = np.arange(np.min(x), np.max(x) + bin_width, bin_width)
    elif isinstance(bins, int):
        edges = np.linspace(np.min(x), np.max(x), bins + 1)
    else:
        raise ValueError("Invalid bins parameter")

    n, _ = np.histogram(x, bins=edges)
    
    # Apply normalization
    if normalization != 'count':
        bin_widths = np.diff(edges)
        if normalization == 'countdensity':
            n = n / bin_widths
        elif normalization == 'cumcount':
            n = np.cumsum(n)
        elif normalization == 'probability':
            n = n / len(x)
        elif normalization == 'percentage':
            n = (100 * n) / len(x)
        elif normalization == 'pdf':
            n = n / (len(x) * bin_widths)
        elif normalization == 'cdf':
            n = np.cumsum(n / len(x))
        else:
            raise ValueError(f"Invalid normalization method: {normalization}")
    
    return n, edges
