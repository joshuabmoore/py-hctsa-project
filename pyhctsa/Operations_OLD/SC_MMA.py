import numpy as np
from warnings import warn 

def SSC_MMA(y, doOverlap = False, scaleRange = None, qRange = None):
    """
    Python implementation of multiscale multifractal analysis (MMA)
    
    Parameters:
    -----------
    y : array_like
        Input time series
    do_overlap : bool, optional
        Whether to use overlapping windows (default is False)
    scale_range : tuple, optional
        (min_scale, max_scale) for analysis (default is None)
    q_range : tuple, optional
        (q_min, q_max) for analysis (default is None)
    
    Returns:
    --------
    dict
        Dictionary containing various MMA statistics
    """

    N = len(y)

    if scaleRange is None:
        scaleRange = [10, np.ceil(N/40)]

    minScale = scaleRange[0]
    maxScale = scaleRange[1]
    if (maxScale/5) < minScale:
        warn(f"Time-series (N={N}) too short for multiscale multifractal analysis")
        out = np.NaN
    elif maxScale % 5 != 0:
        maxScale = np.ceil(maxScale/5)*5
    
    if qRange is None:
        qRange = [-5, 5]
    qMin, qMax = qRange

    qList = np.arange(start=qMin, stop=qMax, step=0.1)
    qList[qList == 0] = 0.0001
    
