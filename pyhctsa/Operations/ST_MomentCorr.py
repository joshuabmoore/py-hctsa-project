import numpy as np
from Operations.IN_MutualInfo import IN_MutualInfo

def ST_MomentCorr(x, windowLength = None, wOverlap = None, mom1 = 'mean', mom2 = 'std', whatTransform = 'none'):
    """
    Correlations between simple statistics in local windows of a time series.
    The idea to implement this was that of Prof. Nick S. Jones (Imperial College London).

    Paramters:
    ----------
    x : array_like
        the input time series
    windowLength : float, optional
        the sliding window length (can be a fraction to specify or a proportion of the time-series length)
    wOverlap : 
        the overlap between consecutive windows as a fraction of the window length
    mom1, mom2 : str, optional
        the statistics to investigate correlations between (in each window):
            (i) 'iqr': interquartile range
            (ii) 'median': median
            (iii) 'std': standard deviation (about the local mean)
            (iv) 'mean': mean
    whatTransform : str, optional
        the pre-processing whatTransformormation to apply to the time series before
        analyzing it:
           (i) 'abs': takes absolute values of all data points
           (ii) 'sqrt': takes the square root of absolute values of all data points
           (iii) 'sq': takes the square of every data point
           (iv) 'none': does no whatTransformormation
    
    Returns:
    --------
    out : dict
        dictionary of statistics related to the correlation between simple statistics in local windows of the input time series. 
    """
    N = len(x) # length of the time series

    if windowLength is None:
        windowLength = 0.02 # 2% of the time-series length
    
    if windowLength < 1:
        windowLength = int(np.ceil(N * windowLength))
    
    # sliding window overlap length
    if wOverlap is None:
        wOverlap = 1/5
    
    if wOverlap < 1:
        wOverlap = int(np.floor(windowLength * wOverlap))

    # Apply the specified whatTransformation
    if whatTransform == 'abs':
        x = np.abs(x)
    elif whatTransform == 'sq':
        x = x**2
    elif whatTransform == 'sqrt':
        x = np.sqrt(np.abs(x))
    elif whatTransform == 'none':
        pass
    else:
        raise ValueError(f"Unknown transformation {whatTransform}")
    
    # create the windows
    x_buff = _buffer(x, windowLength, wOverlap)
    numWindows = (N/(windowLength - wOverlap)) # number of windows

    if np.size(x_buff, 1) > numWindows:
        x_buff = x_buff[:, :-1] # lose the last point

    pointsPerWindow = np.size(x_buff, 0)
    if pointsPerWindow == 1:
        raise ValueError(f"This time series (N = {N}) is too short to extract {numWindows}")
    
    # okay now we have the sliding window ('buffered') signal, x_buff
    # first calculate the first moment in all the windows
    M1 = SUB_CalcMeMoments(x_buff, mom1)
    M2 = SUB_CalcMeMoments(x_buff, mom2)

    out = {}
    rmat = np.corrcoef(M1, M2)
    R = rmat[0, 1] # correlation coeff
    out['R'] = R
    out['absR'] = np.abs(rmat[0, 1])
    out['density'] = np.ptp(M1) * np.ptp(M2) / N
    out['mi'] = IN_MutualInfo(M1, M2, 'gaussian')

    return out
    
# helper functions
def SUB_CalcMeMoments(x_buff, momType):
    if momType == 'mean':
        moms = np.mean(x_buff, axis=0)
    elif momType == 'std':
        moms = np.std(x_buff, axis=0, ddof=1)
    elif momType == 'median':
        moms = np.median(x_buff, axis=0)
    elif momType == 'iqr':
        moms = np.percentile(x_buff, 75, method='hazen', axis=0) - np.percentile(x_buff, 25, method='hazen', axis=0)
    else:
        raise ValueError(f"Unknown statistic {momType}")
    
    return moms
    
def _buffer(X, n, p=0, opt=None):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html.
    Taken from: https://stackoverflow.com/questions/38453249/does-numpy-have-a-function-equivalent-to-matlabs-buffer 

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from X
    '''
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(X):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = X[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), X[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = X[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result
