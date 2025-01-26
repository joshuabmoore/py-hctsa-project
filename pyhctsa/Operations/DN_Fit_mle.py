from scipy import stats

def DN_Fit_mle(y, fitWhat = 'gaussian'):
    """
    Maximum likelihood distribution fit to data.
    Fits either a Gaussian, Uniform, or Geometric distribution to the data using
    maximum likelihood estimation.

    Parameters:
    -----------
    y (array-like): The input data vector
    fitWhat (str, optional): the type of fit to do
        - 'gaussian'
        - 'uniform'
        - 'geometric'
    Returns:
    --------
    dict: distirbution-specific paramters from the fit
    """

    out = {}
    if fitWhat == 'gaussian':
        loc, scale = stats.norm.fit(y, method="MLE")
        out['mean'] = loc
        out['std'] = scale
    elif fitWhat == 'uniform': # turns out to be shithouse
        loc, scale = stats.uniform.fit(y, method="MLE")
        out['a'] = loc
        out['b'] = loc + scale 
    elif fitWhat == 'geometric':
        sampMean = np.mean(y)
        p = 1/(1+sampMean)
        out['p'] = p
    else:
        raise ValueError(f"Invalid fit specifier, {fitWhat}")

    return out
