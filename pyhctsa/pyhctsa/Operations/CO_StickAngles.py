import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from PeripheryFunctions.BF_zscore import BF_zscore as zscore
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.CO_AutoCorr import CO_AutoCorr


def CO_StickAngles(y):
    """
    Analysis of the line-of-sight angles between time series data pts. 

    Line-of-sight angles between time-series pts treat each time-series value as a stick 
    protruding from an opaque baseline level. Statistics are returned on the raw time series, 
    where sticks protrude from the zero-level, and the z-scored time series, where sticks
    protrude from the mean level of the time series.

    Parameters:
    -----------
    y : array-like
        The input time series

    Returns:
    --------
    out : dict
        A dictionary containing various statistics on the obtained sequence of angles.
    """

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
        kde = stats.gaussian_kde(angles[0], bw_method='scott')
        ksy1 = kde(ksx)
        kde2 = stats.gaussian_kde(angles[1], bw_method='scott')
        ksy2 = kde2(ksx)
        out['pnsumabsdiff'] = np.sum(np.abs(ksy1-ksy2))
    else:
        out['pnsumabsdiff'] = np.NaN
    
    # how symmetric is the distribution of angles?
    if len(angles[0]) > 0:
        maxdev = np.max(np.abs(angles[0]))
        kde = stats.gaussian_kde(angles[0], bw_method='scott')
        ksy1 = kde(np.linspace(-maxdev, maxdev, 201))
        #print(ksy1[101:])
        out['symks_p'] = np.sum(np.abs(ksy1[:100] - ksy1[101:][::-1]))
        out['ratmean_p'] = np.mean(angles[0][angles[0] > 0])/np.mean(angles[0][angles[0] < 0])
    else:
        out['symks_p'] = np.NaN
        out['ratmean_p'] = np.NaN
    
    if len(angles[1]) > 0:
        maxdev = np.max(np.abs(angles[1]))
        kde = stats.gaussian_kde(angles[1], bw_method='scott')
        ksy2 = kde(np.linspace(-maxdev, maxdev, 201))
        out['symks_n'] = np.sum(np.abs(ksy2[:100] - ksy2[101:][::-1]))
        out['ratmean_n'] = np.mean(angles[1][angles[1] > 0])/np.mean(angles[1][angles[1] < 0])
    else:
        out['symks_n'] = np.NaN
        out['ratmean_n'] = np.NaN
    
    # z-score
    zangles = []
    zangles.append(zscore(angles[0]))
    zangles.append(zscore(angles[1]))
    zallAngles = zscore(allAngles)

    # how stationary are the angle sets?

    # there are positive angles
    if len(zangles[0]) > 0:
        # StatAv2
        out['statav2_p_m'], out['statav2_p_s'] = SUB_statav(zangles[0], 2)
        # StatAv3
        out['statav3_p_m'], out['statav3_p_s'] = SUB_statav(zangles[0], 3)
        # StatAv4
        out['statav4_p_m'], out['statav4_p_s'] = SUB_statav(zangles[0], 4)
        # StatAv5
        out['statav5_p_m'], out['statav5_p_s'] = SUB_statav(zangles[0], 5)
    else:
        out['statav2_p_m'], out['statav2_p_s'] = np.NaN, np.NaN
        out['statav3_p_m'], out['statav3_p_s'] = np.NaN, np.NaN
        out['statav4_p_m'], out['statav4_p_s'] = np.NaN, np.NaN
        out['statav5_p_m'], out['statav5_p_s'] = np.NaN, np.NaN
    
    # there are negative angles
    if len(zangles[1]) > 0:
        # StatAv2
        out['statav2_n_m'], out['statav2_n_s'] = SUB_statav(zangles[1], 2)
        # StatAv3
        out['statav3_n_m'], out['statav3_n_s'] = SUB_statav(zangles[1], 3)
        # StatAv4
        out['statav4_n_m'], out['statav4_n_s'] = SUB_statav(zangles[1], 4)
        # StatAv5
        out['statav5_n_m'], out['statav5_n_s'] = SUB_statav(zangles[1], 5)
    else:
        out['statav2_n_m'], out['statav2_n_s'] = np.NaN, np.NaN
        out['statav3_n_m'], out['statav3_n_s'] = np.NaN, np.NaN
        out['statav4_n_m'], out['statav4_n_s'] = np.NaN, np.NaN
        out['statav5_n_m'], out['statav5_n_s'] = np.NaN, np.NaN
    
    # All angles
    
    # StatAv2
    out['statav2_all_m'], out['statav2_all_s'] = SUB_statav(zallAngles, 2)
    # StatAv3
    out['statav3_all_m'], out['statav3_all_s'] = SUB_statav(zallAngles, 3)
    # StatAv4
    out['statav4_all_m'], out['statav4_all_s'] = SUB_statav(zallAngles, 4)
    # StatAv5
    out['statav5_all_m'], out['statav5_all_s'] = SUB_statav(zallAngles, 5)
    
    # correlations? 
    if len(zangles[0]) > 0:
        out['tau_p'] = CO_FirstCrossing(zangles[0], 'ac', 0, 'continuous')
        out['ac1_p'] = CO_AutoCorr(zangles[0], 1, 'Fourier')[0]
        out['ac2_p'] = CO_AutoCorr(zangles[0], 2, 'Fourier')[0]
    else:
        out['tau_p'] = np.NaN
        out['ac1_p'] = np.NaN
        out['ac2_p'] = np.NaN
    
    out['tau_all'] = CO_FirstCrossing(zallAngles, 'ac', 0, 'continuous')
    out['ac1_all'] = CO_AutoCorr(zallAngles, 1, 'Fourier')[0]
    out['ac2_all'] = CO_AutoCorr(zallAngles, 2, 'Fourier')[0]


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
            out['skewness_p'], out['kurtosis_p'] = np.NaN, np.Nan, np.NaN,  np.NaN, np.NaN, np.NaN
    
    if len(zangles[1]) > 0:
        out['q1_n'] = np.quantile(zangles[1], 0.01, method='hazen')
        out['q10_n'] = np.quantile(zangles[1], 0.1, method='hazen')
        out['q90_n'] = np.quantile(zangles[1], 0.9, method='hazen')
        out['q99_n'] = np.quantile(zangles[1], 0.99, method='hazen')
        out['skewness_n'] = skew(angles[1])
        out['kurtosis_n'] = kurtosis(angles[1], fisher=False)
    else:
        out['q1_n'], out['q10_n'], out['q90_n'], out['q99_n'], \
            out['skewness_n'], out['kurtosis_n'] = np.NaN, np.NaN, np.NaN,  np.NaN, np.NaN, np.NaN
    
    F_quantz = lambda x : np.quantile(zallAngles, x, method='hazen')
    out['q1_all'] = F_quantz(0.01)
    out['q10_all'] = F_quantz(0.1)
    out['q90_all'] = F_quantz(0.9)
    out['q99_all'] = F_quantz(0.99)
    out['skewness_all'] = skew(allAngles)
    out['kurtosis_all'] = kurtosis(allAngles, fisher=False)



    return out

def SUB_statav(x, n):
    NN = len(x)
    if NN < 2 * n: # not long enough
        statavmean = np.NaN
        statavstd = np.NaN
    x_buff = _buffer(x, int(np.floor(NN/n)))
    if x_buff.shape[1] > n:
        # remove final pt
        x_buff = x_buff[:, :n]
    
    statavmean = np.std(np.mean(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)
    statavstd = np.std(np.std(x_buff, axis=0), ddof=1, axis=0)/np.std(x, ddof=1, axis=0)

    return statavmean, statavstd

def _buffer(X, n, p=0, opt=None):
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
