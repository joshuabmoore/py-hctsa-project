import numpy as np
from Operations.SY_StdNthDer import SY_StdNthDer
from scipy.optimize import curve_fit

def SY_StdNthDerChange(y, maxd = 10):
    """
    How the output of SY_StdNthDer changes with order parameter.
    Order parameter controls the derivative of the signal.

    Operation inspired by a comment on the Matlab Central forum: "You can
    measure the standard deviation of the n-th derivative, if you like." --
    Vladimir Vassilevsky, DSP and Mixed Signal Design Consultant from
    http://www.mathworks.de/matlabcentral/newsreader/view_thread/136539

    Parameters:
    -----------
    y : array-like
        the input time series
    maxd : int, optional
        the maximum derivative to take.

    Returns:
    --------
    out : dict
        the parameters and quality of fit for an exponential model of the variation 
        across successive derivatives

    Note: Uses degree of freedom adjusted RMSE and R2 to align with MATLAB implementation.
    """
    ms = np.array([SY_StdNthDer(y, i) for i in range(1, maxd + 1)])
    # fit exponential growth/decay
    # seed the starting point for params a, b
    p0 = [1, 0.5*np.sign(ms[-1]-ms[0])]
    expFunc = lambda x, a, b : a * np.exp(b*x)
    # fit function using nonlinear least squares
    popt, _ = curve_fit(expFunc, xdata=range(1, maxd+1), ydata=ms, p0=p0, method='lm')
    a, b = popt
    out = {}
    out['fexp_a'] = a 
    out['fexp_b'] = b
    ms_pred = expFunc(range(1, maxd+1), *popt)
    res = ms - ms_pred
    ss_res = np.sum(res**2)
    ss_tot = np.sum((ms - np.mean(ms))**2)
    r_sq = 1 - (ss_res/ss_tot)
    out['fexp_r2'] = r_sq
    out['fexp_adjr2'] = 1 - ((1-r_sq) * (len(ms)-1)) / (len(ms)-len(popt)) # d.o.f adjusted coeff of determination
    # Not mentioned in MATLAB's fitting function that RMSE is actually d.o.f adjusted. Very silly. 
    out['fexp_rmse'] = np.sqrt(np.sum(res**2)/(len(ms)-len(popt)))

    return out
