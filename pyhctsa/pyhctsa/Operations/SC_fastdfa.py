import numpy as np

def SC_fastdfa(x, intervals = None):
    """
    Perform detrended fluctuation analysis on a 
    nonstationary input signal.

    Adapted from the the original fastdfa code by Max A. Little and 
    publicly-available at http://www.maxlittle.net/software/index.php
    M. Little, P. McSharry, I. Moroz, S. Roberts (2006),
    Nonlinear, biophysically-informed speech pathology detection
    in Proceedings of ICASSP 2006, IEEE Publishers: Toulouse, France.

    Parameters:
    -----------
    x: 
        Input signal (must be a 1D numpy array)
    intervals: 
        Optional list of sample interval widths at each scale

    Returns:
    --------
    intervals: 
        List of sample interval widths at each scale
    flucts: 
        List of fluctuation amplitudes at each scale
    """
    if x.ndim != 1:
        raise ValueError("Input sequence must be a vector.")
    
    elements = len(x)
    
    if intervals is None:
        scales = int(np.log2(elements))
        if (1 << (scales - 1)) > elements / 2.5:
            scales -= 1
        intervals = _calculate_intervals(elements, scales)
    else:
        if len(intervals) < 2:
            raise ValueError("Number of intervals must be greater than one.")
        if np.any((intervals > elements) | (intervals < 3)):
            raise ValueError("Invalid interval size: must be between size of sequence x and 3.")
    
    y = np.cumsum(x) # get the cumualtive sum of the input time series
    # perform dfa to get back the flucts at each scale
    flucts = _dfa(y, intervals)
    # now fit a straight line to the log-log plot
    coeffs = np.polyfit(np.log10(intervals), np.log10(flucts), 1)
    alpha = coeffs[0]

    return alpha

# helper functions
def _calculate_intervals(elements, scales):
    # create an array of interval sizes using bitshifting to calculate powers of 2
    return np.array([int((elements / (1 << scale)) + 0.5) for scale in range(scales - 1, -1, -1)])

def _dfa(x, intervals):
    # measure the fluctuations at each scale
    elements = len(x)
    flucts = np.zeros(len(intervals))

    for scale, interval in enumerate(intervals):
        # calculate num subdivisions for this interval size
        subdivs = int(np.ceil(elements / interval))
        trend = np.zeros(elements)

        for i in range(subdivs):
            # calculate start and end indices for current subdivision
            start = i * interval
            end = start + interval
            # if last subdivision extends beyond end of the time series
            if end > elements:
                trend[start:] = x[start:]
                break
            segment = x[start:end]
            # extract the current segment of the detrended time series and fit a linear trend
            t = np.arange(interval)
            coeffs = np.polyfit(t, segment, 1)
            # store the trend values
            trend[start:end] = np.polyval(coeffs, t)
        # compute the root mean square fluctuations for the current interval size, after detrending
        flucts[scale] = np.sqrt(np.mean((x - trend)**2))

    return flucts
