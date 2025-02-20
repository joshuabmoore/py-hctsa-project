from scipy import stats
from sklearn.neighbors import KernelDensity
import numpy as np

def DN_CompareKSFit(x : list, whatDistn : str = 'norm') -> dict:

    # Fit the distribution 'whatDistn' to the input data, x.
    xStep = np.std(x, ddof=1)/100 # set a step size
    if whatDistn == 'norm':
        (a, b) = stats.norm.fit(x)
        peaky = stats.norm.pdf(a, a, b)
        thresh = peaky/100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1, a, b)
        xf2 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2, a, b)
        xf = [xf1, xf2]

    # Estimate smoothed empirical distribution
    kde = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(x[:, None]) # closest match to the default behaviour of ksdensity in MATLAB
    lower = x.min() - 3 * kde.bandwidth_
    upper = x.max() + 3 * kde.bandwidth_
    xi = np.linspace(lower, upper, 100)[:,None]
    f = np.exp(kde.score_samples(xi))
    xi = xi[f > 1E-6]
    if len(xi) == 0:
        return np.nan
    
    xi = [np.floor(xi[0]*10)/10, np.ceil(xi[-1]*10)/10]

    # Find appropriate range [x1 x2] that incorporates the full range of both
    x1 = min(xf[0], xi[0])
    x2 = max(xf[1], xi[1])

    xi = np.linspace(x1, x2, 1000)[:, None]
    kde = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(x[:, None])
    f = np.exp(kde.score_samples(xi))
    ffit = stats.norm.pdf(xi, a, b)
    adiff = np.sum(abs(f - ffit)*(xi[1] - xi[0]))

    return adiff
