import numpy as np

def SY_StdNthDer(y, n = 2):
    """
    Standard deviation of the nth derivative of the time series.

    Based on an idea by Vladimir Vassilevsky, a DSP and Mixed Signal Design
    Consultant in a Matlab forum, who stated that You can measure the standard
    deviation of the nth derivative, if you like".
    cf. http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters:
    -----------
    y : array-like
        the input time series
    n : int, optional
        the order of derivative to analyse

    Returns:
    --------
    out : float
        the std of the nth derivative of the time series
    """

    # crude method of taking a derivative that could be improved upon in future...
    yd = np.diff(y, n=n)
    if len(yd) == 0:
        raise ValueError(f"Time series (N = {len(y)}) too short to compute differences at n = {n}")
    out = np.std(yd, ddof=1)

    return out
