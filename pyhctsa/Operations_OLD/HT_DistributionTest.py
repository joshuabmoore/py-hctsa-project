import numpy as np
from scipy import stats
from scipy.stats import norm, genextreme, uniform, beta, rayleigh, expon, gamma, lognorm, weibull_min

def HT_DistributionTest(x, theTest, theDistn, numBins):
    """
    Hypothesis test for distributional fits to a data vector.

    Fits a distribution to the data and then performs an appropriate hypothesis
    test to quantify the difference between the two distributions.

    Parameters:
    x (array-like): The input data vector
    the_test (str): The hypothesis test to perform:
                    'chi2gof': chi^2 goodness of fit test
                    'ks': Kolmogorov-Smirnov test
                    'lillie': Lilliefors test
    the_distn (str): The distribution to fit:
                     'norm' (Normal)
                     'ev' (Extreme value)
                     'uni' (Uniform)
                     'beta' (Beta)
                     'rayleigh' (Rayleigh)
                     'exp' (Exponential)
                     'gamma' (Gamma)
                     'logn' (Log-normal)
                     'wbl' (Weibull)
    num_bins (int): The number of bins to use for the chi2 goodness of fit test

    Returns:
    float: p-value of the hypothesis test

    """

    # First fit the distribution:
    if theDistn == 'norm':
        params = stats.norm.fit(x)
        dist = stats.norm(*params)
    elif theDistn == 'ev':
        params = stats.gumbel_l.fit(x)
        dist = stats.gumbel_l(*params)
    elif theDistn == 'uni':
        params = stats.uniform.fit(x) # NOTE: returns [loc, loc + scale] instead of [a, b] in MATLAB
        dist = stats.uniform(*params)
    elif theDistn == 'beta':
        # clumsily scale to the range (0,1)
        x = (x - np.min(x) + 0.01*np.std(x, ddof=1)) / (np.max(x) - np.min(x) + 0.01*np.std(x, ddof=1))
        params = stats.beta.fit(x)
        



