import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.ST_SimpleStats import ST_SimpleStats

def ST_LocalExrema(y, howToWindow = 'l', n = None):
    """
    How local maximums and minimums vary across the time series.

    Finds maximums and minimums within given segments of the time series and
    analyzes the results.

    Parameters:
    -----------
    y : array-like
        The input time series
    howToWindow : str, optional 
        Method to determine window size
        'l': windows of a given length
        'n': a specified number of windows to break the time series up into
        'tau': sets a window length equal to the correlation length of the time series
    n : int, optional
        Specifies the window length or number of windows, depending on howToWindow

    Returns:
    --------
    dict: 
        A dictionary containing various statistics about local extrema
    """
    if n is None:
        if howToWindow == 'l':
            n = 100 # 100 sample windows
        elif howToWindow == 'n':
            n = 5 # 5 windows
    
    N = len(y)

    # Set the window length
    if howToWindow == 'l':
        windowLength = n # window length
    elif howToWindow == 'n':
        windowLength = int(np.floor(N/n))
    elif howToWindow == 'tau':
        windowLength = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    else:
        raise ValueError(f"Unknown method {howToWindow}")
    
    if (windowLength > N) or (windowLength <= 1):
        # This feature is unsuitable if the window length exceeds ts
        out = np.NaN
    
    # Buffer the time series
    y_buff = _buffer(y, windowLength) # no overlap
    # each column is a window of samples
    if np.all(y_buff[:, -1] == 0):
        y_buff = y_buff[:, :-1]  # remove last window if zero-padded

    numWindows = np.size(y_buff, 1) # number of windows

    # Find local extrema
    locMax = np.max(y_buff, axis=0) # summary of local maxima
    locMin = np.min(y_buff, axis=0) # summary of local minima
    absLocMin = np.abs(locMin) # abs val of local minima
    exti = np.where(absLocMin > locMax)
    loc_ext = locMax.copy()
    loc_ext[exti] = locMin[exti] # local extrema (furthest from mean; either maxs or mins)
    abs_loc_ext = np.abs(loc_ext) # the magnitude of the most extreme events in each window

    # Return Outputs
    out = {
        'meanrat': np.mean(locMax) / np.mean(absLocMin),
        'medianrat': np.median(locMax) / np.median(absLocMin),
        'minmax': np.min(locMax),
        'minabsmin': np.min(absLocMin),
        'minmaxonminabsmin': np.min(locMax) / np.min(absLocMin),
        'meanmax': np.mean(locMax),
        'meanabsmin': np.mean(absLocMin),
        'meanext': np.mean(loc_ext),
        'medianmax': np.median(locMax),
        'medianabsmin': np.median(absLocMin),
        'medianext': np.median(loc_ext),
        'stdmax': np.std(locMax, ddof=1),
        'stdmin': np.std(locMin, ddof=1),
        'stdext': np.std(loc_ext, ddof=1),
        'zcext': np.sum(np.diff(np.sign(loc_ext)) != 0) / (numWindows - 1),  # zero crossings
        'meanabsext': np.mean(abs_loc_ext),
        'medianabsext': np.median(abs_loc_ext),
        'diffmaxabsmin': np.sum(np.abs(locMax - absLocMin)) / numWindows,
        'uord': np.sum(np.sign(loc_ext)) / numWindows,
        'maxmaxmed': np.max(locMax) / np.median(locMax),
        'minminmed': np.min(locMin) / np.median(locMin),
        'maxabsext': np.max(abs_loc_ext) / np.median(abs_loc_ext)
    }


    return out

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
