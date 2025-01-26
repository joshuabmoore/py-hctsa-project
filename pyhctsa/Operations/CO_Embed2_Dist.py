from scipy.stats import expon
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_AutoCorr import CO_AutoCorr
from numpy import histogram_bin_edges
import numpy as np

def CO_Embed2_Dist(y, tau = None):
    """
    Analyzes distances in a 2-dim embedding space of a time series.

    Returns statistics on the sequence of successive Euclidean distances between
    points in a two-dimensional time-delay embedding space with a given
    time-delay, tau.

    Outputs include the autocorrelation of distances, the mean distance, the
    spread of distances, and statistics from an exponential fit to the
    distribution of distances.

    Parameters:
    y (array-like): A z-scored column vector representing the input time series.
    tau (int, optional): The time delay. If None, it's set to the first minimum of the autocorrelation function.

    Returns:
    dict: A dictionary containing various statistics of the embedding.
    """

    N = len(y) # time-series length

    if tau is None:
        tau = 'tau' # set to the first minimum of autocorrelation function
    
    if tau == 'tau':
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
        if tau > N / 10:
            tau = N//10

    # Make sure the time series is a column vector
    y = np.asarray(y).reshape(-1, 1)

    # Construct a 2-dimensional time-delay embedding (delay of tau)
    m = np.hstack((y[:-tau], y[tau:]))

    # Calculate Euclidean distances between successive points in this space, d:
    out = {}
    d = np.sqrt(np.sum(np.diff(m, axis=0)**2, axis=1))
    
    # Calculate autocorrelations
    out['d_ac1'] = CO_AutoCorr(d, 1, 'Fourier')[0] # lag 1 ac
    out['d_ac2'] = CO_AutoCorr(d, 2, 'Fourier')[0] # lag 2 ac
    out['d_ac3'] = CO_AutoCorr(d, 3, 'Fourier')[0] # lag 3 ac

    out['d_mean'] = np.mean(d) # Mean distance
    out['d_median'] = np.median(d) # Median distance
    out['d_std'] = np.std(d, ddof=1) # Standard deviation of distances
    # need to use Hazen method of computing percentiles to get IQR consistent with MATLAB
    q75 = np.percentile(d, 75, method='hazen')
    q25 = np.percentile(d, 25, method='hazen')
    iqr_val = q75 - q25
    out['d_iqr'] = iqr_val # Interquartile range of distances
    out['d_max'] = np.max(d) # Maximum distance
    out['d_min'] = np.min(d) # Minimum distance
    out['d_cv'] = np.mean(d) / np.std(d, ddof=1) # Coefficient of variation of distances

    # Empirical distances distribution often fits Exponential distribution quite well
    # Fit to all values (often some extreme outliers, but oh well)
    l = 1 / np.mean(d)
    nlogL = -np.sum(expon.logpdf(d, scale=1/l))
    out['d_expfit_nlogL'] = nlogL

    # Calculate histogram
    # unable to get exact equivalence with MATLAB's histcount function, altough numpy's histogram_edges gets very close...
    N, bin_edges = histcounts_(d, bins='auto', normalization='probability')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #print(bin_edges)
    exp_fit = expon.pdf(bin_centers, scale=1/l)
    out['d_expfit_meandiff'] = np.mean(np.abs(N - exp_fit))

    return out

# helper function
def histcounts_(x, bins=None, binEdges=None, normalization='probability'):
    x = np.asarray(x).flatten()
    if binEdges is not None:
        edges = np.asarray(binEdges)
    elif bins is None or bins == 'auto':
        edges = histogram_bin_edges(x, bins='auto')
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
