import numpy as np

def MD_rawHRVmeas(x):
    """
    MD_rawHRVmeas computes Poincare plot measures used in HRV (Heart Rate Variability) analysis.
    
    This function computes the triangular histogram index and Poincare plot measures for a time
    series assumed to measure sequences of consecutive RR intervals in milliseconds. It is not 
    suitable for other types of time series.

    Parameters:
    ----------
    x : array_like
        A time series assumed to measure sequences of consecutive RR intervals in milliseconds.

    Returns:
    -------
    out : dict
        A dictionary containing the following keys:
        - 'tri10': Triangular histogram index with 10 bins
        - 'tri20': Triangular histogram index with 20 bins
        - 'trisqrt': Triangular histogram index with bins calculated using the square root method
        - 'SD1': Standard deviation of the Poincare plot's minor axis
        - 'SD2': Standard deviation of the Poincare plot's major axis
    """

    N = len(x)
    out = {}

    # triangular histogram index
    hist_counts10, _ = np.histogram(x, 10)
    out['tri10'] = N/np.max(hist_counts10)
    hist_counts20, _ = np.histogram(x, 20)
    out['tri20'] = N/np.max(hist_counts20)
    # MATLAB histcounts returns wrong number of bins for sqrt rule. This should be the correct num of bins...
    hist_counts_sqrt, _ = np.histogram(x, bins=int(np.ceil(np.sqrt(N))))
    out['trisqrt'] = N/np.max(hist_counts_sqrt)

    # Poincare plot measures
    diffx = np.diff(x)
    out['SD1'] = 1/np.sqrt(2) * np.std(diffx, ddof=1) * 1000
    out['SD2'] = np.sqrt(2 * np.var(x, ddof=1) - (1/2) * np.std(diffx, ddof=1)**2) * 1000

    return out
