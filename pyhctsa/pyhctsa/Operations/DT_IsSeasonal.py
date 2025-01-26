import numpy as np
from scipy.optimize import curve_fit

def DT_IsSeasonal(y):
    """
    A simple test of seasonality.
    """

    N = len(y) # length of the time series
    # IMPORTANT: need to start at index 1 if want fitting to give
    # the same results as MATLAB
    r = np.arange(1, N+1) # range over which to fit

    # Fit a sinusoidal model
    smodel = lambda x, a1, b1, c1 : a1 * np.sin(b1*x+c1)
    


    
