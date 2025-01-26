import numpy as np

def CO_NonlinearAutoCorr(y, taus, doAbs=None):
    """
    A custom nonlinear autocorrelation of a time series.

    Nonlinear autocorrelations are of the form:
    <x_i x_{i-tau_1} x{i-tau_2}...>
    The usual two-point autocorrelations are
    <x_i.x_{i-tau}>

    Parameters:
    y (array-like): Should be the z-scored time series (Nx1 vector)
    taus (array-like): Should be a vector of the time delays (mx1 vector)
        e.g., [2] computes <x_i x_{i-2}>
        e.g., [1,2] computes <x_i x_{i-1} x_{i-2}>
        e.g., [1,1,3] computes <x_i x_{i-1}^2 x_{i-3}>
        e.g., [0,0,1] computes <x_i^3 x_{i-1}>
    do_abs (bool, optional): If True, takes an absolute value before taking the final mean.
        Useful for an odd number of contributions to the sum.
        Default is to do this for odd numbers anyway, if not specified.

    Returns:
    out (float): The computed nonlinear autocorrelation.

    Notes:
    - For odd numbers of regressions (i.e., even number length taus vectors)
      the result will be near zero due to fluctuations below the mean;
      even for highly-correlated signals. (do_abs)
    - do_abs = True is really a different operation that can't be compared with
      the values obtained from taking do_abs = False (i.e., for odd lengths of taus)
    - It can be helpful to look at nonlinearAC at each iteration.
    """
    if doAbs == None:
        if len(taus) % 2 == 1:
            doAbs = 0
        else:
            doAbs = 1

    N = len(y)
    tmax = np.max(taus)

    nlac = y[tmax:N]

    for i in taus:
        nlac = np.multiply(nlac,y[ tmax - i:N - i ])

    if doAbs:
        out = np.mean(np.absolute(nlac))

    else:
        out = np.mean(nlac)

    return out
