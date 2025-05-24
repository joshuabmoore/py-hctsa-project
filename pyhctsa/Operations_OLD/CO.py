import numpy as np
from numpy import histogram_bin_edges
from loguru import logger
from statsmodels.tsa.stattools import pacf
from Operations_OLD.BF_PointOfCrossing import BF_PointOfCrossing # replace
from Operations_OLD.BF_iszscored import BF_iszscored # replace
from Operations_OLD.BF_zscore import BF_zscore # replace
from Operations_OLD.BF_SignChange import BF_SignChange
from Operations_OLD.BF_MutualInformation import BF_MutualInformation # replace
from typing import Union
from scipy.optimize import curve_fit
from scipy.stats import mode as smode
from scipy.stats import expon, skew, kurtosis, gaussian_kde
from Operations_OLD.binpicker import binpicker


def StickAngles(y : list) -> dict:
    """
    Analysis of the line-of-sight angles between time series data pts. 

    Line-of-sight angles between time-series pts treat each time-series value as a stick 
    protruding from an opaque baseline level. Statistics are returned on the raw time series, 
    where sticks protrude from the zero-level, and the z-scored time series, where sticks
    protrude from the mean level of the time series.

    Note: Any features derived from the KDE will be different to the output from MATLAB 
    as a result of differences in how the KDE works in scipy vs MATLAB.
    This will affect the features: `pnsumabsdiff`, `symks_p`, `symks_n`, `ratmean_p`, `ratmean_n`.

    Parameters:
    -----------
    y : array-like
        The input time series

    Returns:
    --------
    out : dict
        A dictionary containing various statistics on the obtained sequence of angles.
    """
    y = np.array(y)
    # Split the time series into positive and negative parts
    ix = [np.where(y >= 0)[0], np.where(y < 0)[0]]
    n = [len(ix[0]), len(ix[1])]

    # Compute the stick angles
    angles = [[], []]
    for j in range(2):
        if n[j] > 1:
            diff_y = np.diff(y[ix[j]])
            diff_x = np.diff(ix[j])
            angles[j] = np.arctan(diff_y /diff_x)
    allAngles = np.concatenate(angles)

    # Initialise output dictionary
    out = {}
    out['std_p'] = np.nanstd(angles[0], ddof=1) 
    out['mean_p'] = np.nanmean(angles[0]) 
    out['median_p'] = np.nanmedian(angles[0])

    out['std_n'] = np.nanstd(angles[1], ddof=1)
    out['mean_n'] = np.nanmean(angles[1])
    out['median_n'] = np.nanmedian(angles[1])

    out['std'] = np.nanstd(allAngles, ddof=1)
    out['mean'] = np.nanmean(allAngles)
    out['median'] = np.nanmedian(allAngles)

    # difference between positive and negative angles
    # return difference in densities
    ksx = np.linspace(np.min(allAngles), np.max(allAngles), 200)
    if len(angles[0]) > 0 and len(angles[1]) > 0:
        kde = gaussian_kde(angles[0], bw_method='scott')
        ksy1 = kde(ksx)
        kde2 = gaussian_kde(angles[1], bw_method='scott')
        ksy2 = kde2(ksx)
        out['pnsumabsdiff'] = np.sum(np.abs(ksy1-ksy2))
    else:
        out['pnsumabsdiff'] = np.nan
    
    # how symmetric is the distribution of angles?
    if len(angles[0]) > 0:
        maxdev = np.max(np.abs(angles[0]))
        kde = gaussian_kde(angles[0], bw_method='scott')
        ksy1 = kde(np.linspace(-maxdev, maxdev, 201))
        #print(ksy1[101:])
        out['symks_p'] = np.sum(np.abs(ksy1[:100] - ksy1[101:][::-1]))
        out['ratmean_p'] = np.mean(angles[0][angles[0] > 0])/np.mean(angles[0][angles[0] < 0])
    else:
        out['symks_p'] = np.nan
        out['ratmean_p'] = np.nan
    
    if len(angles[1]) > 0:
        maxdev = np.max(np.abs(angles[1]))
        kde = gaussian_kde(angles[1], bw_method='scott')
        ksy2 = kde(np.linspace(-maxdev, maxdev, 201))
        out['symks_n'] = np.sum(np.abs(ksy2[:100] - ksy2[101:][::-1]))
        out['ratmean_n'] = np.mean(angles[1][angles[1] > 0])/np.mean(angles[1][angles[1] < 0])
    else:
        out['symks_n'] = np.nan
        out['ratmean_n'] = np.nan
    
    # z-score
    zangles = []
    zangles.append(BF_zscore(angles[0]))
    zangles.append(BF_zscore(angles[1]))
    zallAngles = BF_zscore(allAngles)

    # how stationary are the angle sets?

    # there are positive angles
    if len(zangles[0]) > 0:
        # StatAv2
        out['statav2_p_m'], out['statav2_p_s'] = _SUB_statav(zangles[0], 2)
        # StatAv3
        out['statav3_p_m'], out['statav3_p_s'] = _SUB_statav(zangles[0], 3)
        # StatAv4
        out['statav4_p_m'], out['statav4_p_s'] = _SUB_statav(zangles[0], 4)
        # StatAv5
        out['statav5_p_m'], out['statav5_p_s'] = _SUB_statav(zangles[0], 5)
    else:
        out['statav2_p_m'], out['statav2_p_s'] = np.nan, np.nan
        out['statav3_p_m'], out['statav3_p_s'] = np.nan, np.nan
        out['statav4_p_m'], out['statav4_p_s'] = np.nan, np.nan
        out['statav5_p_m'], out['statav5_p_s'] = np.nan, np.nan
    
    # there are negative angles
    if len(zangles[1]) > 0:
        # StatAv2
        out['statav2_n_m'], out['statav2_n_s'] = _SUB_statav(zangles[1], 2)
        # StatAv3
        out['statav3_n_m'], out['statav3_n_s'] = _SUB_statav(zangles[1], 3)
        # StatAv4
        out['statav4_n_m'], out['statav4_n_s'] = _SUB_statav(zangles[1], 4)
        # StatAv5
        out['statav5_n_m'], out['statav5_n_s'] = _SUB_statav(zangles[1], 5)
    else:
        out['statav2_n_m'], out['statav2_n_s'] = np.nan, np.nan
        out['statav3_n_m'], out['statav3_n_s'] = np.nan, np.nan
        out['statav4_n_m'], out['statav4_n_s'] = np.nan, np.nan
        out['statav5_n_m'], out['statav5_n_s'] = np.nan, np.nan
    
    # All angles
    
    # StatAv2
    out['statav2_all_m'], out['statav2_all_s'] = _SUB_statav(zallAngles, 2)
    # StatAv3
    out['statav3_all_m'], out['statav3_all_s'] = _SUB_statav(zallAngles, 3)
    # StatAv4
    out['statav4_all_m'], out['statav4_all_s'] = _SUB_statav(zallAngles, 4)
    # StatAv5
    out['statav5_all_m'], out['statav5_all_s'] = _SUB_statav(zallAngles, 5)
    
    # correlations? 
    if len(zangles[0]) > 0:
        out['tau_p'] = FirstCrossing(zangles[0], 'ac', 0, 'continuous')
        out['ac1_p'] = AutoCorr(zangles[0], 1, 'Fourier')[0]
        out['ac2_p'] = AutoCorr(zangles[0], 2, 'Fourier')[0]
    else:
        out['tau_p'] = np.nan
        out['ac1_p'] = np.nan
        out['ac2_p'] = np.nan
    
    if len(zangles[1]) > 0:
        out['tau_n'] = FirstCrossing(zangles[1], 'ac', 0, 'continuous')
        out['ac1_n'] = AutoCorr(zangles[1], 1, 'Fourier')[0]
        out['ac2_n'] = AutoCorr(zangles[1], 2, 'Fourier')[0]
    else:
        out['tau_n'] = np.nan
        out['ac1_n'] = np.nan
        out['ac2_n'] = np.nan
    
    out['tau_all'] = FirstCrossing(zallAngles, 'ac', 0, 'continuous')
    out['ac1_all'] = AutoCorr(zallAngles, 1, 'Fourier')[0]
    out['ac2_all'] = AutoCorr(zallAngles, 2, 'Fourier')[0]


    # What does the distribution look like? 
    
    # Some quantiles and moments
    if len(zangles[0]) > 0:
        out['q1_p'] = np.quantile(zangles[0], 0.01, method='hazen')
        out['q10_p'] = np.quantile(zangles[0], 0.1, method='hazen')
        out['q90_p'] = np.quantile(zangles[0], 0.9, method='hazen')
        out['q99_p'] = np.quantile(zangles[0], 0.99, method='hazen')
        out['skewness_p'] = skew(angles[0])
        out['kurtosis_p'] = kurtosis(angles[0], fisher=False)
    else:
        out['q1_p'], out['q10_p'], out['q90_p'], out['q99_p'], \
            out['skewness_p'], out['kurtosis_p'] = np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan
    
    if len(zangles[1]) > 0:
        out['q1_n'] = np.quantile(zangles[1], 0.01, method='hazen')
        out['q10_n'] = np.quantile(zangles[1], 0.1, method='hazen')
        out['q90_n'] = np.quantile(zangles[1], 0.9, method='hazen')
        out['q99_n'] = np.quantile(zangles[1], 0.99, method='hazen')
        out['skewness_n'] = skew(angles[1])
        out['kurtosis_n'] = kurtosis(angles[1], fisher=False)
    else:
        out['q1_n'], out['q10_n'], out['q90_n'], out['q99_n'], \
            out['skewness_n'], out['kurtosis_n'] = np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan
    
    F_quantz = lambda x : np.quantile(zallAngles, x, method='hazen')
    out['q1_all'] = F_quantz(0.01)
    out['q10_all'] = F_quantz(0.1)
    out['q90_all'] = F_quantz(0.9)
    out['q99_all'] = F_quantz(0.99)
    out['skewness_all'] = skew(allAngles)
    out['kurtosis_all'] = kurtosis(allAngles, fisher=False)

    return out

def _SUB_statav(x, n):
    # helper function
    NN = len(x)
    if NN < 2 * n: # not long enough
        statavmean = np.nan
        statavstd = np.nan
    x_buff = _buffer(x, int(np.floor(NN/n)))
    if x_buff.shape[1] > n:
        # remove final pt
        x_buff = x_buff[:, :n]
    
    statavmean = np.std(np.mean(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)
    statavstd = np.std(np.std(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)

    return statavmean, statavstd

def _buffer(X, n, p=0, opt=None):
    # helper function
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

def RM_AMInformation(y : list, tau : int = 1):
    """
    A wrapper for rm_information(), which calculates automutal information

    Inputs:
        y, the input time series
        tau, the time lag at which to calculate automutal information

    :returns estimate of mutual information

    - Wrapper initially developed by Ben D. Fulcher in MATLAB
    - rm_information.py initially developed by Rudy Moddemeijer in MATLAB
    - Translated to python by Tucker Cullen

    """

    if tau >= len(y):
        return np.nan
    elif tau == 0:
        # handle the case when tau = 0 (no lag)
        y1 = y2 = y
    else:
        y1 = y[:-tau]
        y2 = y[tau:]

    out = _RM_information(y1, y2)

    return out[0]


def _RM_histogram2(*args):
    # helper function
    """
    rm_histogram2() computes the two dimensional frequency histogram of two row vectors x and y

    Takes in either two or three parameters:
        rm_histogram(x, y)
        rm_histogram(x, y, descriptor)

    x, y : the row vectors to be analyzed
    descriptor : the descriptor of the histogram where:

        descriptor = [lowerx, upperx, ncellx, lowery, uppery, ncelly]
            lower? : the lowerbound of the ? dimension of the histogram
            upper? : the upperbound of the dimension of the histogram
            ncell? : the number of cells of the ? dimension of the histogram

    :return: a tuple countaining a) the result (the 2d frequency histogram), b) descriptor (the descriptor used)

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen

    """

    nargin = len(args)

    if nargin < 1:
        print("Usage: result = rm_histogram2(X, Y)")
        print("       result = rm_histogram2(X,Y)")
        print("Where: descriptor = [lowerX, upperX, ncellX; lowerY, upperY, ncellY")

    # some initial tests on the input arguments

    x = np.array(args[0])  # make sure the imputs are in numpy array form
    y = np.array(args[1])

    xshape = x.shape
    yshape = y.shape

    lenx = xshape[0]  # how many elements are in the row vector
    leny = yshape[0]

    if len(xshape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(yshape) != 1:
        print("Error: invalid dimension of y")
        return

    if lenx != leny:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if nargin > 3:
        print("Error: too many arguments")
        return

    if nargin == 2:
        minx = np.amin(x)
        maxx = np.amax(x)
        deltax = (maxx - minx) / (lenx - 1)
        ncellx = np.ceil(lenx ** (1 / 3))

        miny = np.amin(y)
        maxy = np.amax(y)
        deltay = (maxy - miny) / (leny - 1)
        ncelly = ncellx
        descriptor = np.array(
            [[minx - deltax / 2, maxx + deltax / 2, ncellx], [miny - deltay / 2, maxy + deltay / 2, ncelly]])
    else:
        descriptor = args[2]

    lowerx = descriptor[0, 0]  # python indexes one less then matlab indexes, since starts at zero
    upperx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowery = descriptor[1, 0]
    uppery = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    # checking descriptor to make sure it is valid, otherwise print an error

    if ncellx < 1:
        print("Error: invalid number of cells in X dimension")

    if ncelly < 1:
        print("Error: invalid number of cells in Y dimension")

    if upperx <= lowerx:
        print("Error: invalid bounds in X dimension")

    if uppery <= lowery:
        print("Error: invalid bounds in Y dimension")

    result = np.zeros([int(ncellx), int(ncelly)],
                      dtype=int)  # should do the same thing as matlab: result(1:ncellx,1:ncelly) = 0;

    xx = np.around((x - lowerx) / (upperx - lowerx) * ncellx + 1 / 2)
    yy = np.around((y - lowery) / (uppery - lowery) * ncelly + 1 / 2)

    xx = xx.astype(int)  # cast all the values in xx and yy to ints for use in indexing, already rounded in previous step
    yy = yy.astype(int)

    for n in range(0, lenx):
        indexx = xx[n]
        indexy = yy[n]

        indexx -= 1  # adjust indices to start at zero, not one like in MATLAB
        indexy -= 1

        if indexx >= 0 and indexx <= ncellx - 1 and indexy >= 0 and indexy <= ncelly - 1:
            result[indexx, indexy] = result[indexx, indexy] + 1

    return result, descriptor

def _RM_information(*args):
    # helper function
    """
    rm_information estimates the mutual information of the two stationary signals with
    independent pairs of samples using various approaches:

    takes in between 2 and 5 parameters:
        rm_information(x, y)
        rm_information(x, y, descriptor)
        rm_information(x, y, descriptor, approach)
        rm_information(x, y, descriptor, approach, base)

    :returns estimate, nbias, sigma, descriptor

        estimate : the mututal information estimate
        nbias : n-bias of the estimate
        sigma : the standard error of the estimate
        descriptor : the descriptor of the histogram, see also rm_histogram2

            lowerbound? : lowerbound of the histogram in the ? direction
            upperbound? : upperbound of the histogram in the ? direction
            ncell? : number of cells in the histogram in ? direction

        approach : method used, choose from the following:

            'unbiased'  : the unbiased estimate (default)
            'mmse'      : minimum mean square estimate
            'biased'    : the biased estimate

        base : the base of the logarithm, default e

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen
    """

    nargin = len(args)

    if nargin < 1:
        print("Takes in 2-5 parameters: ")
        print("rm_information(x, y)")
        print("rm_information(x, y, descriptor)")
        print("rm_information(x, y, descriptor, approach)")
        print("rm_information(x, y, descriptor, approach, base)")
        print()

        print("Returns a tuple containing: ")
        print("estimate, nbias, sigma, descriptor")
        return

    # some initial tests on the input arguments

    x = np.array(args[0])  # make sure the imputs are in numpy array form
    y = np.array(args[1])

    xshape = x.shape
    yshape = y.shape

    lenx = xshape[0]  # how many elements are in the row vector
    leny = yshape[0]

    if len(xshape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(yshape) != 1:
        print("Error: invalid dimension of y")
        return

    if lenx != leny:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if nargin > 5:
        print("Error: too many arguments")
        return

    if nargin < 2:
        print("Error: not enough arguments")
        return

    # setting up variables depending on amount of inputs

    if nargin == 2:
        hist = _RM_histogram2(x, y)  # call outside function from rm_histogram2.py
        h = hist[0]
        descriptor = hist[1]

    if nargin >= 3:
        hist = _RM_histogram2(x, y, args[2])  # call outside function from rm_histogram2.py, args[2] represents the given descriptor
        h = hist[0]
        descriptor = hist[1]

    if nargin < 4:
        approach = 'unbiased'
    else:
        approach = args[3]

    if nargin < 5:
        base = np.e  # as in e = 2.71828
    else:
        base = args[4]

    lowerboundx = descriptor[0, 0]  #not sure why most of these were included in the matlab script, most of them go unused
    upperboundx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowerboundy = descriptor[1, 0]
    upperboundy = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    estimate = 0
    sigma = 0
    count = 0

    # determine row and column sums

    hy = np.sum(h, 0)
    hx = np.sum(h, 1)

    ncellx = ncellx.astype(int)
    ncelly = ncelly.astype(int)

    for nx in range(0, ncellx):
        for ny in range(0, ncelly):
            if h[nx, ny] != 0:
                logf = np.log(h[nx, ny] / hx[nx] / hy[ny])
            else:
                logf = 0

            count = count + h[nx, ny]
            estimate = estimate + h[nx, ny] * logf
            sigma = sigma + h[nx, ny] * (logf ** 2)

    # biased estimate

    estimate = estimate / count
    sigma = np.sqrt((sigma / count - estimate ** 2) / (count - 1))
    estimate = estimate + np.log(count)
    nbias = (ncellx - 1) * (ncelly - 1) / (2 * count)

    # conversion to unbiased estimate

    if approach[0] == 'u':
        estimate = estimate - nbias
        nbias = 0

        # conversion to minimum mse estimate

    if approach[0] == 'm':
        estimate = estimate - nbias
        nbias = 0
        lamda = (estimate ** 2) / ((estimate ** 2) + (sigma ** 2))
        nbias = (1 - lamda) * estimate
        estimate = lamda * estimate
        sigma = lamda * sigma

        # base transformations

    estimate = estimate / np.log(base)
    nbias = nbias / np.log(base)
    sigma = sigma / np.log(base)

    return estimate, nbias, sigma, descriptor

def FZCGLSCF(y : list, alpha : Union[float, int], beta : Union[float, int], maxtau : Union[int, None] = None) -> float:
    """
    Compute the first zero-crossing of the generalized self-correlation function (GLSCF).

    This function returns the first zero-crossing point of the GLSCF, as introduced by
    Duarte Queiros and Moyano (2007) in *Physica A*, Vol. 383, pp. 10–15, titled:
    "Yet on statistical properties of traded volume: Correlation and mutual information
    at different value magnitudes."

    The function uses the `GLSCF` metric to compute generalized self-correlations over
    increasing lags and detects the first zero-crossing via linear interpolation.

    Parameters
    ----------
    y : list
        The input time series.
    alpha : float
        The `alpha` parameter of the GLSCF.
    beta : float
        The `beta` parameter of the GLSCF.
    maxtau : int, optional
        The maximum time lag to evaluate (default is the length of `y`).

    Returns
    -------
    float
        The estimated first zero-crossing lag of the GLSCF. If no zero-crossing is found
        within `maxtau`, returns `maxtau`.

    Notes
    -----
    This function requires the `GLSCF(y, alpha, beta, tau)` function to be defined.
    """
    N = len(y) # the length of the time series

    if maxtau is None:
        maxtau = N
    
    glscfs = np.zeros(maxtau)

    for i in range(1, maxtau+1):
        tau = i

        glscfs[i-1] = GLSCF(y, alpha, beta, tau)
        if (i > 1) and (glscfs[i-1]*glscfs[i-2] < 0):
            # Draw a straight line between these two and look at where it hits zero
            out = i - 1 + glscfs[i-1]/(glscfs[i-1]-glscfs[i-2])
            return out
    
    return maxtau # if the function hasn't exited yet, set output to maxtau 

def GLSCF(y : list, alpha : Union[float, int], beta : Union[float, int], tau : Union[str, int, None] = 'tau') -> float:
    """
    Compute the generalized linear self-correlation function (GLSCF) of a time series.

    This function was introduced in Queirós and Moyano (2007) in *Physica A*, 
    Vol. 383, pp. 10–15: 
    "Yet on statistical properties of traded volume: Correlation and mutual information 
    at different value magnitudes."
    [https://doi.org/10.1016/j.physa.2007.04.068]

    The GLSCF captures correlations in the magnitude of a time series, generalizing 
    the traditional autocorrelation function to emphasize specific relationships between 
    values depending on their scale.

    Parameters
    ----------
    y : list or array_like
        The input time series.
    alpha : float
        A non-zero exponent applied to the earlier time point in the correlation.
    beta : float
        A non-zero exponent applied to the later time point in the correlation.
    tau : int
        The time delay (lag) at which to compute the GLSCF.

    Returns
    -------
    float
        The GLSCF value at lag `tau`.

    Notes
    -----
    - When `alpha == beta`, the function emphasizes correlations between values 
      of similar magnitude.
    - When `alpha != beta`, it emphasizes interactions between different magnitudes.
    - The function reduces to the traditional autocorrelation when `alpha = beta = 1`.
    """
    # Set tau to first zero-crossing of the autocorrelation function with the input 'tau'
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Take magnitudes of time-delayed versions of the time series
    y1 = np.abs(y[:-tau])
    y2 = np.abs(y[tau:])


    p1 = np.mean(np.multiply((y1 ** alpha), (y2 ** beta)))
    p2 = np.multiply(np.mean(y1 ** alpha), np.mean(y2 ** beta))
    p3 = np.sqrt(np.mean(y1 ** (2*alpha)) - (np.mean(y1 ** alpha))**2)
    p4 = np.sqrt(np.mean(y2 ** (2*beta)) - (np.mean(y2 ** beta))**2)

    glscf = (p1 - p2) / (p3 * p4)

    return glscf    

def Embed2(y : list, tau : Union[str, int, None] = 'tau'):
    """
    Statistics of the time series in a 2-dimensional embedding space.

    Embeds the (z-scored) time series in a two-dimensional time-delay
    embedding space with a given time-delay, tau, and outputs a set of
    statistics about the structure in this space, including angular
    distribution, etc.

    Parameters:
    y (array-like): The input time series (will be converted to a column vector)
    tau (int or str, optional): The time-delay. If 'tau', it will be set to the first zero-crossing of ACF.

    Returns:
    dict: A dictionary containing various statistics about the embedded time series
    """

    # Set tau to the first zero-crossing of the autocorrelation function with the 'tau' input
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
        if tau > len(y) / 10:
            tau = len(y) // 10
    # Ensure that y is a column vector
    y = np.array(y).reshape(-1, 1)

    # Construct the two-dimensional recurrence space
    m = np.hstack((y[:-tau], y[tau:]))
    N = m.shape[0] # number of points in the recurrence space
    

    # 1) Distribution of angles time series; angles between successive points in this space
    theta = np.divide(np.diff(m[:, 1]), np.diff(m[:, 0]))
    theta = np.arctan(theta) # measured as deviation from the horizontal

    out = {}

    out['theta_ac1'] = AutoCorr(theta, 1, 'Fourier')[0]
    out['theta_ac2'] = AutoCorr(theta, 2, 'Fourier')[0]
    out['theta_ac3'] = AutoCorr(theta, 3, 'Fourier')[0]

    out['theta_mean'] = np.mean(theta)
    out['theta_std'] = np.std(theta, ddof=1)
    
    binEdges = np.linspace(-np.pi/2, np.pi/2, 11) # 10 bins in the histogram
    px, _ = _histcounts(theta, binEdges=binEdges, normalization='probability')
    binWidths = np.diff(binEdges)
    out['hist10std'] = np.std(px, ddof=1)
    out['histent'] = -np.sum(px[px>0] * np.log(px[px>0] / binWidths[px>0]))
    

    # Stationarity in fifths of the time series
    # Use histograms with 4 bins
    x = np.linspace(-np.pi/2, np.pi/2, 5) # 4 bins
    afifth = (N-1) // 5 # -1 because angles are correlations *between* points
    n = np.zeros((len(x)-1, 5))
    for i in range(5):
        n[:, i], _ = np.histogram(theta[afifth*i:afifth*(i+1)], bins=x)
        
    n = n / afifth
    
    for i in range(4):
        out[f'stdb{i+1}'] = np.std(n[:, i], ddof=1)

    # STATIONARITY of points in the space (do they move around in the space)
    # (1) in terms of distance from origin
    afifth = N // 5
    buffer_m = [m[afifth*i:afifth*(i+1), :] for i in range(5)]

    # Mean euclidean distance in each segment
    eucdm = [np.mean(np.sqrt(x[:, 0]**2 + x[:, 1]**2)) for x in buffer_m]
    for i in range(5):
        out[f'eucdm{i+1}'] = eucdm[i]
    out['std_eucdm'] = np.std(eucdm, ddof=1)
    out['mean_eucdm'] = np.mean(eucdm)

    # Standard deviation of Euclidean distances in each segment
    eucds = [np.std(np.sqrt(x[:, 0]**2 + x[:, 1]**2), ddof=1) for x in buffer_m]
    for i in range(5):
        out[f'eucds{i+1}'] = eucds[i]
    out['std_eucds'] = np.std(eucds, ddof=1)
    out['mean_eucds'] = np.mean(eucds)

    # Maximum volume in each segment (defined as area of rectangle of max span in each direction)
    maxspanx = [np.ptp(x[:, 0]) for x in buffer_m]
    maxspany = [np.ptp(x[:, 1]) for x in buffer_m]
    spanareas = np.multiply(maxspanx, maxspany)
    out['stdspana'] = np.std(spanareas, ddof=1)
    out['meanspana'] = np.mean(spanareas)

    # Outliers in the embedding space
    # area of max span of all points; versus area of max span of 50% of points closest to origin
    d = np.sqrt(m[:, 0]**2 + m[:, 1]**2)
    ix = np.argsort(d)
    
    out['areas_all'] = np.ptp(m[:, 0]) * np.ptp(m[:, 1])
    r50 = ix[:int(np.ceil(len(ix)/2))] # ceil to match MATLAB's round fn output
    
    out['areas_50'] = np.ptp(m[r50, 0]) * np.ptp(m[r50, 1])
    out['arearat'] = out['areas_50'] / out['areas_all']

    return out 

def Embed2_Shapes(y : list, tau : Union[str, int, None] = 'tau', shape : str = 'circle', r : float = 1.0) -> dict:
    """
    Shape-based statistics in a 2-d embedding space.

    Takes a shape and places it on each point in the two-dimensional time-delay
    embedding space sequentially. This function counts the points inside this shape
    as a function of time, and returns statistics on this extracted time series.

    Parameters:
    -----------
    y : array_like
        The input time-series as a (z-scored) column vector.
    tau : int or str, optional
        The time-delay. If 'tau', it's set to the first zero crossing of the autocorrelation function.
    shape : str, optional
        The shape to use. Currently only 'circle' is supported.
    r : float, optional
        The radius of the circle.

    Returns:
    --------
    dict
        A dictionary containing various statistics of the constructed time series.
    """
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
        # cannot set time delay > 10% of the length of the time series...
        if tau > len(y)/10:
            tau = int(np.floor(len(y)/10))
        
    # Create the recurrence space, populated by points m
    m = np.column_stack((y[:-tau], y[tau:]))
    N = len(m)

    # Start the analysis
    counts = np.zeros(N)
    if shape == 'circle':
        # Puts a circle around each point in the embedding space in turn
        # counts how many pts are inside this shape, looks at the time series thus formed
        for i in range(N): # across all pts in the time series
            m_c = m - m[i] # pts wrt current pt i
            m_c_d = np.sum(m_c**2, axis=1) # Euclidean distances from pt i
            counts[i] = np.sum(m_c_d <= r**2) # number of pts enclosed in a circle of radius r
    else:
        raise ValueError(f"Unknown shape '{shape}'")
    
    counts -= 1 # ignore self counts

    if np.all(counts == 0):
        print("No counts detected!")
        return np.nan

    # Return basic statistics on the counts
    out = {}
    out['ac1'] = AutoCorr(counts, 1, 'Fourier')[0]
    out['ac2'] = AutoCorr(counts, 2, 'Fourier')[0]
    out['ac3'] = AutoCorr(counts, 3, 'Fourier')[0]
    out['tau'] = FirstCrossing(counts, 'ac', 0, 'continuous')
    out['max'] = np.max(counts)
    out['std'] = np.std(counts, ddof=1)
    out['median'] = np.median(counts)
    out['mean'] = np.mean(counts)
    out['iqr'] = np.percentile(counts, 75, method='hazen') - np.percentile(counts, 25, method='hazen')
    out['iqronrange'] = out['iqr']/np.ptp(counts)

    # distribution - using sqrt binning method
    numBinsToUse = int(np.ceil(np.sqrt(len(counts)))) # supposed to be what MATLAB uses for 'sqrt' option.
    minX, maxX = np.min(counts), np.max(counts)
    binWidthEst = (maxX - minX)/numBinsToUse
    edges = binpicker(minX, maxX, None, binWidthEst) # mimics the functionality of MATLAB's internal function for selecting bins
    binCounts, binEdges = np.histogram(counts, bins=edges)
    # normalise bin counts
    binCountsNorm = np.divide(binCounts, np.sum(binCounts))
    # get bin centres
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    out['mode_val'] = np.max(binCountsNorm)
    out['mode'] = binCentres[np.argmax(binCountsNorm)]
    # histogram entropy
    out['hist_ent'] = np.sum(binCountsNorm[binCountsNorm > 0] * np.log(binCountsNorm[binCountsNorm > 0]))

    # Stationarity measure for fifths of the time series
    afifth = int(np.floor(N/5))
    buffer_m = np.array([counts[i*afifth:(i+1)*afifth] for i in range(5)])
    out['statav5_m'] = np.std(np.mean(buffer_m, axis=1), ddof=1) / np.std(counts, ddof=1)
    out['statav5_s'] = np.std(np.std(buffer_m, axis=1, ddof=1), ddof=1) / np.std(counts, ddof=1)

    return out

def Embed2_Dist(y : list, tau : Union[None, str] = None):
    """
    Analyzes distances in a 2-dim embedding space of a time series.

    Returns statistics on the sequence of successive Euclidean distances between
    points in a two-dimensional time-delay embedding space with a given
    time-delay, tau.

    Outputs include the autocorrelation of distances, the mean distance, the
    spread of distances, and statistics from an exponential fit to the
    distribution of distances.

    Parameters:
    y (array-like): A z-scored column vector representing the input time series.
    tau (int, optional): The time delay. If None, it's set to the first minimum of the autocorrelation function.

    Returns:
    dict: A dictionary containing various statistics of the embedding.
    """

    N = len(y) # time-series length

    if tau is None:
        tau = 'tau' # set to the first minimum of autocorrelation function
    
    if tau == 'tau':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
        if tau > N / 10:
            tau = N//10

    # Make sure the time series is a column vector
    y = np.asarray(y).reshape(-1, 1)

    # Construct a 2-dimensional time-delay embedding (delay of tau)
    m = np.hstack((y[:-tau], y[tau:]))

    # Calculate Euclidean distances between successive points in this space, d:
    out = {}
    d = np.sqrt(np.sum(np.diff(m, axis=0)**2, axis=1))
    
    # Calculate autocorrelations
    out['d_ac1'] = AutoCorr(d, 1, 'Fourier')[0] # lag 1 ac
    out['d_ac2'] = AutoCorr(d, 2, 'Fourier')[0] # lag 2 ac
    out['d_ac3'] = AutoCorr(d, 3, 'Fourier')[0] # lag 3 ac

    out['d_mean'] = np.mean(d) # Mean distance
    out['d_median'] = np.median(d) # Median distance
    out['d_std'] = np.std(d, ddof=1) # Standard deviation of distances
    # need to use Hazen method of computing percentiles to get IQR consistent with MATLAB
    q75 = np.percentile(d, 75, method='hazen')
    q25 = np.percentile(d, 25, method='hazen')
    iqr_val = q75 - q25
    out['d_iqr'] = iqr_val # Interquartile range of distances
    out['d_max'] = np.max(d) # Maximum distance
    out['d_min'] = np.min(d) # Minimum distance
    out['d_cv'] = np.mean(d) / np.std(d, ddof=1) # Coefficient of variation of distances

    # Empirical distances distribution often fits Exponential distribution quite well
    # Fit to all values (often some extreme outliers, but oh well)
    l = 1 / np.mean(d)
    nlogL = -np.sum(expon.logpdf(d, scale=1/l))
    out['d_expfit_nlogL'] = nlogL

    # Calculate histogram
    # unable to get exact equivalence with MATLAB's histcount function, although numpy's histogram_edges gets very close...
    N, bin_edges = _histcounts(d, bins='auto', normalization='probability')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #print(bin_edges)
    exp_fit = expon.pdf(bin_centers, scale=1/l)
    out['d_expfit_meandiff'] = np.mean(np.abs(N - exp_fit))

    return out

# helper function
def _histcounts(x, bins=None, binEdges=None, normalization='probability'):
    x = np.asarray(x).flatten()
    if binEdges is not None:
        edges = np.asarray(binEdges)
    elif bins is None or bins == 'auto':
        edges = histogram_bin_edges(x, bins='auto')
    elif isinstance(bins, int):
        edges = np.linspace(np.min(x), np.max(x), bins + 1)
    else:
        raise ValueError("Invalid bins parameter")

    n, _ = np.histogram(x, bins=edges)
    
    # Apply normalization
    if normalization != 'count':
        bin_widths = np.diff(edges)
        if normalization == 'countdensity':
            n = n / bin_widths
        elif normalization == 'cumcount':
            n = np.cumsum(n)
        elif normalization == 'probability':
            n = n / len(x)
        elif normalization == 'percentage':
            n = (100 * n) / len(x)
        elif normalization == 'pdf':
            n = n / (len(x) * bin_widths)
        elif normalization == 'cdf':
            n = np.cumsum(n / len(x))
        else:
            raise ValueError(f"Invalid normalization method: {normalization}")
    
    return n, edges

def Embed2_Basic(y : list, tau : int = 1):
    """
    Point density statistics in a 2-d embedding space.

    Computes a set of point-density statistics in a plot of y_i against y_{i-tau}.

    Parameters:
    -----------
    y : array_like
        The input time series.
    tau : int or str, optional
        The time lag (can be set to 'tau' to set the time lag to the first zero
        crossing of the autocorrelation function).

    Returns:
    --------
    out : dict
        Dictionary containing various point density statistics.
    """

    if tau == 'tau':
        # Make tau the first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')

    xt = y[:-tau]  # part of the time series
    xtp = y[tau:]  # time-lagged time series
    N = len(y) - tau  # Length of each time series subsegment

    out = {}

    # Points in a thick bottom-left -- top-right diagonal
    out['updiag01'] = np.sum(np.abs(xtp - xt) < 0.1) / N
    out['updiag05'] = np.sum(np.abs(xtp - xt) < 0.5) / N

    # Points in a thick bottom-right -- top-left diagonal
    out['downdiag01'] = np.sum(np.abs(xtp + xt) < 0.1) / N
    out['downdiag05'] = np.sum(np.abs(xtp + xt) < 0.5) / N

    # Ratio of these
    out['ratdiag01'] = out['updiag01'] / out['downdiag01']
    out['ratdiag05'] = out['updiag05'] / out['downdiag05']

    # In a thick parabola concave up
    out['parabup01'] = np.sum(np.abs(xtp - xt**2) < 0.1) / N
    out['parabup05'] = np.sum(np.abs(xtp - xt**2) < 0.5) / N

    # In a thick parabola concave down
    out['parabdown01'] = np.sum(np.abs(xtp + xt**2) < 0.1) / N
    out['parabdown05'] = np.sum(np.abs(xtp + xt**2) < 0.5) / N

    # In a thick parabola concave up, shifted up 1
    out['parabup01_1'] = np.sum(np.abs(xtp - (xt**2 + 1)) < 0.1) / N
    out['parabup05_1'] = np.sum(np.abs(xtp - (xt**2 + 1)) < 0.5) / N

    # In a thick parabola concave down, shifted up 1
    out['parabdown01_1'] = np.sum(np.abs(xtp + (xt**2 - 1)) < 0.1) / N
    out['parabdown05_1'] = np.sum(np.abs(xtp + (xt**2 - 1)) < 0.5) / N

    # In a thick parabola concave up, shifted down 1
    out['parabup01_n1'] = np.sum(np.abs(xtp - (xt**2 - 1)) < 0.1) / N
    out['parabup05_n1'] = np.sum(np.abs(xtp - (xt**2 - 1)) < 0.5) / N

    # In a thick parabola concave down, shifted down 1
    out['parabdown01_n1'] = np.sum(np.abs(xtp + (xt**2 + 1)) < 0.1) / N
    out['parabdown05_n1'] = np.sum(np.abs(xtp + (xt**2 + 1)) < 0.5) / N

    # RINGS (points within a radius range)
    out['ring1_01'] = np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.1) / N
    out['ring1_02'] = np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.2) / N
    out['ring1_05'] = np.sum(np.abs(xtp**2 + xt**2 - 1) < 0.5) / N

    # CIRCLES (points inside a given circular boundary)
    out['incircle_01'] = np.sum(xtp**2 + xt**2 < 0.1) / N
    out['incircle_02'] = np.sum(xtp**2 + xt**2 < 0.2) / N
    out['incircle_05'] = np.sum(xtp**2 + xt**2 < 0.5) / N
    out['incircle_1'] = np.sum(xtp**2 + xt**2 < 1) / N
    out['incircle_2'] = np.sum(xtp**2 + xt**2 < 2) / N
    out['incircle_3'] = np.sum(xtp**2 + xt**2 < 3) / N
    
    incircle_values = [out['incircle_01'], out['incircle_02'], out['incircle_05'],
                       out['incircle_1'], out['incircle_2'], out['incircle_3']]
    out['medianincircle'] = np.median(incircle_values)
    out['stdincircle'] = np.std(incircle_values, ddof=1)
    
    return out

def Embed2_AngleTau(y : list, maxTau : int):
    """
    Angle autocorrelation in a 2-dimensional embedding space.

    Investigates how the autocorrelation of angles between successive points in
    the two-dimensional time-series embedding change as tau varies from
    tau = 1, 2, ..., maxTau.

    Parameters:
    -----------
    y (numpy.ndarray): Input time series (1D array)
    maxTau (int): The maximum time lag to consider

    Returns:
    --------
    dict: A dictionary containing various statistics
    """
    tauRange = np.arange(1, maxTau + 1)
    numTau = len(tauRange)

    # Ensure y is a column vector
    y = np.atleast_2d(y)
    if y.shape[0] < y.shape[1]:
        y = y.T

    stats_store = np.zeros((3, numTau))

    for i, tau in enumerate(tauRange):
        m = np.column_stack((y[:-tau], y[tau:]))
        theta = np.diff(m[:, 1]) / np.diff(m[:, 0])
        theta = np.arctan(theta)  # measured as deviation from the horizontal

        if len(theta) == 0:
            raise ValueError(f'Time series (N={len(y)}) too short for embedding')

        stats_store[0, i] = AutoCorr(theta, 1, 'Fourier')[0]
        stats_store[1, i] = AutoCorr(theta, 2, 'Fourier')[0]
        stats_store[2, i] = AutoCorr(theta, 3, 'Fourier')[0]
    
    # Compute output statistics
    out = {
        'ac1_thetaac1': AutoCorr(stats_store[0, :], 1, 'Fourier'),
        'ac1_thetaac2': AutoCorr(stats_store[1, :], 1, 'Fourier'),
        'ac1_thetaac3': AutoCorr(stats_store[2, :], 1, 'Fourier'),
        'mean_thetaac1': np.mean(stats_store[0, :]),
        'max_thetaac1': np.max(stats_store[0, :]),
        'min_thetaac1': np.min(stats_store[0, :]),
        'mean_thetaac2': np.mean(stats_store[1, :]),
        'max_thetaac2': np.max(stats_store[1, :]),
        'min_thetaac2': np.min(stats_store[1, :]),
        'mean_thetaac3': np.mean(stats_store[2, :]),
        'max_thetaac3': np.max(stats_store[2, :]),
        'min_thetaac3': np.min(stats_store[2, :]),
    }

    out['meanrat_thetaac12'] = out['mean_thetaac1'] / out['mean_thetaac2']
    out['diff_thetaac12'] = np.sum(np.abs(stats_store[1, :] - stats_store[0, :]))

    return out


def CompareMinAMI(y : list, binMethod : str, numBins : int = 10) -> dict:
    """
    Variability in first minimum of automutual information.

    Finds the first minimum of the automutual information by various different
    estimation methods, and sees how this varies over different coarse-grainings
    of the time series.

    Args:
    y (array-like): The input time series
    binMethod (str): The method for estimating mutual information (input to CO_HistogramAMI)
    numBins (int or array-like): The number of bins for the AMI estimation to compare over

    Returns:
    dict: A dictionary containing various statistics on the set of first minimums 
          of the automutual information function
    """
    N = len(y)
    # Range of time lags to consider
    tauRange = np.arange(0, int(np.ceil(N/2))+1)
    numTaus = len(tauRange)

    # range of bin numbers to consider
    if isinstance(numBins, int):
        numBins = [numBins]
    
    numBinsRange = len(numBins)
    amiMins = np.zeros(numBinsRange)

    # Calculate automutual information
    for i in range(numBinsRange):  # vary over number of bins in histogram
        amis = np.zeros(numTaus)
        for j in range(numTaus):  # vary over time lags, tau
            amis[j] = HistogramAMI(y, tauRange[j], binMethod, numBins[i])
            if (j > 1) and ((amis[j] - amis[j-1]) * (amis[j-1] - amis[j-2]) < 0):
                amiMins[i] = tauRange[j-1]
                break
        if amiMins[i] == 0:
            amiMins[i] = tauRange[-1]
    # basic statistics
    out = {}
    out['min'] = np.min(amiMins)
    out['max'] = np.max(amiMins)
    out['range'] = np.ptp(amiMins)
    out['median'] = np.median(amiMins)
    out['mean'] = np.mean(amiMins)
    out['std'] = np.std(amiMins, ddof=1) # will return NaN for single values instead of 0
    out['nunique'] = len(np.unique(amiMins))
    out['mode'], out['modef'] = smode(amiMins)
    out['modef'] = out['modef']/numBinsRange

    # converged value? 
    out['conv4'] = np.mean(amiMins[-5:])

    # look for peaks (local maxima)
    # % local maxima above 1*std from mean
    # inspired by curious result of periodic maxima for periodic signal with
    # bin size... ('quantiles', [2:80])
    diff_ami_mins = np.diff(amiMins[:-1])
    positive_diff_indices = np.where(diff_ami_mins > 0)[0]
    sign_change_indices = BF_SignChange(diff_ami_mins, 1)

    # Find the intersection of positive_diff_indices and sign_change_indices
    loc_extr = np.intersect1d(positive_diff_indices, sign_change_indices) + 1
    above_threshold_indices = np.where(amiMins > out['mean'] + out['std'])[0]
    big_loc_extr = np.intersect1d(above_threshold_indices, loc_extr)

    # Count the number of elements in big_loc_extr
    out['nlocmax'] = len(big_loc_extr)

    return out

def AutoCorrShape(y : list, stopWhen : Union[int, str] = 'posDrown'):
    """
    How the autocorrelation function changes with the time lag.

    Outputs include the number of peaks, and autocorrelation in the
    autocorrelation function (ACF) itself.

    Parameters:
    -----------
    y : array_like
        The input time series
    stopWhen : str or int, optional
        The criterion for the maximum lag to measure the ACF up to.
        Default is 'posDrown'.

    Returns:
    --------
    dict
        A dictionary containing various metrics about the autocorrelation function.
    """
    N = len(y)

    # Only look up to when two consecutive values are under the significance threshold
    th = 2 / np.sqrt(N)  # significance threshold

    # Calculate the autocorrelation function, up to a maximum lag, length of time series (hopefully it's cropped by then)
    acf = []

    # At what lag does the acf drop to zero, Ndrown (by my definition)?
    if isinstance(stopWhen, int):
        taus = list(range(0, stopWhen+1))
        acf = AutoCorr(y, taus, 'Fourier')
        Ndrown = stopWhen
        
    elif stopWhen in ['posDrown', 'drown', 'doubleDrown']:
        # Compute ACF up to a given threshold:
        Ndrown = 0 # the point at which ACF ~ 0
        if stopWhen == 'posDrown':
            # stop when ACF drops below threshold, th
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0]
                if np.isnan(acf_val):
                    logger.warning("Weird time series (constant?)")
                    out = np.nan
                if acf_val < th:
                    # Ensure ACF is all positive
                    if acf_val > 0:
                        Ndrown = i
                        acf.append(acf_val)
                    else:
                        # stop at the previous point if not positive
                        Ndrown = i-1
                    # ACF has dropped below threshold, break the for loop...
                    break
                # hasn't dropped below thresh, append to list 
                acf.append(acf_val)
            # This should yield the initial, positive portion of the ACF.
            assert all(np.array(acf) > 0)
        elif stopWhen == 'drown':
            # Stop when ACF is very close to 0 (within threshold, th = 2/sqrt(N))
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0] # acf vector indicies are not lags
                # if positive and less than thresh
                if i > 0 and abs(acf_val) < th:
                    Ndrown = i
                    acf.append(acf_val)
                    break
                acf.append(acf_val)
        elif stopWhen == 'doubleDrown':
            # Stop at 2*tau, where tau is the lag where ACF ~ 0 (within 1/sqrt(N) threshold)
            for i in range(1, N+1):
                acf_val = AutoCorr(y, i-1, 'Fourier')[0]
                if Ndrown > 0 and i == Ndrown * 2:
                    acf.append(acf_val)
                    break
                elif i > 1 and abs(acf_val) < th:
                    Ndrown = i
                acf.append(acf_val)
    else:
        raise ValueError(f"Unknown ACF decay criterion: '{stopWhen}'")

    acf = np.array(acf)
    Nac = len(acf)

    # Check for good behavior
    if np.any(np.isnan(acf)):
        # This is an anomalous time series (e.g., all constant, or conatining NaNs)
        out = np.NaN
    
    out = {}
    out['Nac'] = Ndrown

    # Basic stats on the ACF
    out['sumacf'] = np.sum(acf)
    out['meanacf'] = np.mean(acf)
    if stopWhen != 'posDrown':
        out['meanabsacf'] = np.mean(np.abs(acf))
        out['sumabsacf'] = np.sum(np.abs(acf))

    # Autocorrelation of the ACF
    minPointsForACFofACF = 5 # can't take lots of complex stats with fewer than this

    if Nac > minPointsForACFofACF:
        out['ac1'] = AutoCorr(acf, 1, 'Fourier')[0]
        if all(acf > 0):
            out['actau'] = np.nan
        else:
            out['actau'] = AutoCorr(acf, FirstCrossing(acf, 'ac', 0, 'discrete'), 'Fourier')[0]

    else:
        out['ac1'] = np.nan
        out['actau'] = np.nan
    
    # Local extrema
    dacf = np.diff(acf)
    ddacf = np.diff(dacf)
    extrr = BF_SignChange(dacf, 1)
    sdsp = ddacf[extrr]

    # Proportion of local minima
    out['nminima'] = np.sum(sdsp > 0)
    out['meanminima'] = np.mean(sdsp[sdsp > 0])

    # Proportion of local maxima
    out['nmaxima'] = np.sum(sdsp < 0)
    out['meanmaxima'] = abs(np.mean(sdsp[sdsp < 0])) # must be negative: make it positive

    # Proportion of extrema
    out['nextrema'] = len(sdsp)
    out['pextrema'] = len(sdsp) / Nac

    # Fit exponential decay (only for 'posDrown', and if there are enough points)
    # Should probably only do this up to the first zero crossing...
    fitSuccess = False
    minPointsToFitExp = 4 # (need at least four points to fit exponential)

    if stopWhen == 'posDrown' and Nac >= minPointsToFitExp:
        # Fit exponential decay to (absolute) ACF:
        # (kind of only makes sense for the first positive period)
        expFunc = lambda x, b : np.exp(-b * x)
        try:
            popt, _ = curve_fit(expFunc, np.arange(Nac), acf, p0=0.5)
            fitSuccess = True
        except:
            fitSuccess = False
        
    if fitSuccess:
        bFit = popt[0] # fitted b
        out['decayTimescale'] = 1 / bFit
        expFit = expFunc(np.arange(Nac), bFit)
        residuals = acf - expFit
        out['fexpacf_r2'] = 1 - (np.sum(residuals**2) / np.sum((acf - np.mean(acf))**2))
        # had to fit a second exponential function with negative b to get same output as MATLAB for std residuals
        expFit2 = expFunc(np.arange(Nac), -bFit)
        residuals2 = acf - expFit2
        out['fexpacf_stdres'] = np.std(residuals2, ddof=1) # IMPORTANT *** DDOF=1 TO MATCH MATLAB STD ***

    else:
        # Fit inappropriate (or failed): return NaNs for the relevant stats
        out['decayTimescale'] = np.nan
        out['fexpacf_r2'] = np.nan
        out['fexpacf_stdres'] = np.nan
    
    return out

def NonlinearAutoCorr(y : list, taus, doAbs = None):
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
        nlac = np.multiply(nlac,y[tmax - i:N - i])

    if doAbs:
        out = np.mean(np.absolute(nlac))

    else:
        out = np.mean(nlac)

    return out


def HistogramAMI(y, tau = 1, meth = 'even', numBins = 10):
    """
    CO_HistogramAMI: The automutual information of the distribution using histograms.

    Parameters:
    y (array-like): The input time series
    tau (int, list or str): The time-lag(s) (default: 1)
    meth (str): The method of computing automutual information:
                'even': evenly-spaced bins through the range of the time series,
                'std1', 'std2': bins that extend only up to a multiple of the
                                standard deviation from the mean of the time series to exclude outliers,
                'quantiles': equiprobable bins chosen using quantiles.
    num_bins (int): The number of bins (default: 10)

    Returns:
    float or dict: The automutual information calculated in this way.
    """
    # Use first zero crossing of the ACF as the time lag
    if isinstance(tau, str) and tau in ['ac', 'tau']:
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    
    # Bins for the data
    # same for both -- assume same distribution (true for stationary processes, or small lags)
    if meth == 'even':
        b = np.linspace(np.min(y), np.max(y), numBins + 1)
        # Add increment buffer to ensure all points are included
        inc = 0.1
        b[0] -= inc
        b[-1] += inc
    elif meth == 'std1': # bins out to +/- 1 std
        b = np.linspace(-1, 1, numBins + 1)
        if np.min(y) < -1:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 1:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'std2': # bins out to +/- 2 std
        b = np.linspace(-2, 2, numBins + 1)
        if np.min(y) < -2:
            b = np.concatenate(([np.min(y) - 0.1], b))
        if np.max(y) > 2:
            b = np.concatenate((b, [np.max(y) + 0.1]))
    elif meth == 'quantiles': # use quantiles with ~equal number in each bin
        b = np.quantile(y, np.linspace(0, 1, numBins + 1), method='hazen')
        b[0] -= 0.1
        b[-1] += 0.1
    else:
        raise ValueError(f"Unknown method '{meth}'")
    
    # Sometimes bins can be added (e.g., with std1 and std2), so need to redefine numBins
    numBins = len(b) - 1

    # Form the time-delay vectors y1 and y2
    if not isinstance(tau, (list, np.ndarray)):
        # if only single time delay as integer, make into a one element list
        tau = [tau]

    amis = np.zeros(len(tau))

    for i, t in enumerate(tau):
        if t == 0:
            # for tau = 0, y1 and y2 are identical to y
            y1 = y2 = y
        else:
            y1 = y[:-t]
            y2 = y[t:]
        # Joint distribution of y1 and y2
        pij, _, _ = np.histogram2d(y1, y2, bins=(b, b))
        pij = pij[:numBins, :numBins]  # joint
        pij = pij / np.sum(pij)  # normalize
        pi = np.sum(pij, axis=1)  # marginal
        pj = np.sum(pij, axis=0)  # other marginal

        pii = np.tile(pi, (numBins, 1)).T
        pjj = np.tile(pj, (numBins, 1))

        r = pij > 0  # Defining the range in this way, we set log(0) = 0
        amis[i] = np.sum(pij[r] * np.log(pij[r] / pii[r] / pjj[r]))

    if len(tau) == 1:
        return amis[0]
    else:
        return {f'ami{i+1}': ami for i, ami in enumerate(amis)}


def FirstCrossing(y : list, corr_fun : str ='ac', threshold : float = 0.0, what_out='both'):
    """
    The first crossing of a given autocorrelation across a given threshold.

    Parameters:
    -----------
    y : array_like
        The input time series
    corr_fun : str, optional
        The self-correlation function to measure:
        'ac': normal linear autocorrelation function
    threshold : float, optional
        Threshold to cross. Examples: 0 [first zero crossing], 1/np.e [first 1/e crossing]
    what_out : str, optional
        Specifies the output format: 'both', 'discrete', or 'continuous'

    Returns:
    --------
    out : dict or float
        The first crossing information, format depends on what_out
    """
    # Select the self-correlation function
    if corr_fun == 'ac':
        # Autocorrelation at all time lags
        corrs = AutoCorr(y, [], 'Fourier')
    else:
        raise ValueError(f"Unknown correlation function '{corr_fun}'")

    # Calculate point of crossing
    first_crossing_index, point_of_crossing_index = BF_PointOfCrossing(corrs, threshold)

    # Assemble the appropriate output (dictionary or float)
    # Convert from index space (1,2,…) to lag space (0,1,2,…)
    if what_out == 'both':
        out = {
            'firstCrossing': first_crossing_index - 1,
            'pointOfCrossing': point_of_crossing_index - 1
        }
    elif what_out == 'discrete':
        out = first_crossing_index - 1
    elif what_out == 'continuous':
        out = point_of_crossing_index - 1
    else:
        raise ValueError(f"Unknown output format '{what_out}'")

    return out

def AutoCorr(y : list, tau : int = 1, method : str = 'Fourier'):
    """
    Compute the autocorrelation of an input time series.

    Parameters:
    -----------
    y : array_like
        A scalar time series column vector.
    tau : int, list, optional
        The time-delay. If tau is a scalar, returns autocorrelation for y at that
        lag. If tau is a list, returns autocorrelations for y at that set of
        lags. If empty list, returns the full function for the 'Fourier' estimation method.
    method : str, optional
        The method of computing the autocorrelation: 'Fourier',
        'TimeDomainStat', or 'TimeDomain'.

    Returns:
    --------
    out : float or array
        The autocorrelation at the given time lag(s).

    Notes:
    ------
    Specifying method = 'TimeDomain' can tolerate NaN values in the time
    series.
    """
    N = len(y)  # time-series length

    if tau:
        # if list is not empty
        if np.max(tau) > N - 1:  # -1 because acf(1) is lag 0
            logger.warning(f"Time lag {np.max(tau)} is too long for time-series length {N}.")
        if np.any(np.array(tau) < 0):
            logger.warning('Negative time lags not applicable.')
    
    if method == 'Fourier':
        n_fft = 2 ** (int(np.ceil(np.log2(N))) + 1)
        F = np.fft.fft(y - np.mean(y), n_fft)
        F = F * np.conj(F)
        acf = np.fft.ifft(F)  # Wiener–Khinchin
        acf = acf / acf[0]  # Normalize
        acf = np.real(acf)
        acf = acf[:N]
        
        if not tau:  # list empty, return the full function
            out = acf
        else:  # return a specific set of values
            tau = np.atleast_1d(tau)
            out = np.zeros(len(tau))
            for i, t in enumerate(tau):
                if (t > len(acf) - 1) or (t < 0):
                    out[i] = np.nan
                else:
                    out[i] = acf[t]
    
    elif method == 'TimeDomainStat':
        sigma2 = np.std(y, ddof=1)**2  # time-series variance
        mu = np.mean(y)  # time-series mean
        
        def acf_y(t):
            return np.mean((y[:N-t] - mu) * (y[t:] - mu)) / sigma2
        
        tau = np.atleast_1d(tau)
        out = np.array([acf_y(t) for t in tau])
    
    elif method == 'TimeDomain':
        tau = np.atleast_1d(tau)
        out = np.zeros(len(tau))
        
        for i, t in enumerate(tau):
            if np.any(np.isnan(y)):
                good_r = (~np.isnan(y[:N-t])) & (~np.isnan(y[t:]))
                print(f'NaNs in time series, computing for {np.sum(good_r)}/{len(good_r)} pairs of points')
                y1 = y[:N-t]
                y1n = y1[good_r] - np.mean(y1[good_r])
                y2 = y[t:]
                y2n = y2[good_r] - np.mean(y2[good_r])
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean(y1n * y2n) / np.std(y1[good_r], ddof=0) / np.std(y2[good_r], ddof=0)
            else:
                y1 = y[:N-t]
                y2 = y[t:]
                # std() ddof adjusted to be consistent with numerator's N normalization
                out[i] = np.mean((y1 - np.mean(y1)) * (y2 - np.mean(y2))) / np.std(y1, ddof=0) / np.std(y2, ddof=0)
    
    else:
        raise ValueError(f"Unknown autocorrelation estimation method '{method}'.")
    
    return out

def PartialAutoCorr(y, max_tau=10, what_method='ols'):
    """
    Compute the partial autocorrelation of an input time series.

    Parameters:
    ----------
    y (array-like): A scalar time series column vector.
    max_tau (int): The maximum time-delay. Returns for lags up to this maximum.
    what_method (str): The method used to compute: 'ols' or 'yw' (Yule-Walker).

    Returns:
    ----------
    out (dict): The partial autocorrelations across the set of time lags.

    Raises:
    ----------
    ValueError: If max_tau is negative or what_method is invalid.
    """
    y = np.array(y)
    N = len(y)  # time-series length

    if max_tau <= 0:
        raise ValueError('Negative or zero time lags not applicable')

    method_map = {'ols': 'ols', 'Yule-Walker': 'ywm'} 
    if what_method not in method_map:
        raise ValueError(f"Invalid method: {what_method}. Use 'ols' or 'Yule-Walker'.")

    # Compute partial autocorrelation
    pacf_values = pacf(y, nlags=max_tau, method=method_map[what_method])

    # Create output dictionary
    out = {}
    for i in range(1, max_tau + 1):
        out[f'pac_{i}'] = pacf_values[i]

    return out
