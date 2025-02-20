# scaling
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline
import statsmodels.api as sm
from CO import AutoCorr
from loguru import logger


def FluctAnal(x, q = 2, wtf = 'rsrange', tauStep = 1, k = 1, lag = None, logInc = True):
    """
    """

    N = len(x) # time series length

    # Compute integrated sequence
    if lag is None or lag == 1:
        y = np.cumsum(x) # normal cumulative sum
    else:
        y = np.cumsum(x[::lag]) # if a lag is specified, do a decimation
    
    # Perform scaling over a range of tau, up to a fifth the time-series length
    #-------------------------------------------------------------------------------
    # Peng (1995) suggests 5:N/4 for DFA
    # Caccia suggested from 10 to (N-1)/2...
    #-------------------------------------------------------------------------------

    if logInc:
        # in this case tauStep is the number of points to compute
        if tauStep == 1:
            # handle the case where tauStep is 1, but we want to take the upper 
            # limit (MATLAB) rather than the lower (Python)
            tauStep += 1
            logRange = np.linspace(np.log(5), np.log(np.floor(N/2)), tauStep)[1] # take the second entry (upper limit)
        else:
            logRange = np.linspace(np.log(5), np.log(np.floor(N/2)), tauStep)
        taur = np.unique(np.round(np.exp(logRange)).astype(int))
    else:
        taur = np.arange(5, int(np.floor(N/2)) + 1, tauStep)
    ntau = len(taur) # % analyze the time series across this many timescales
    #print(taur)
    if ntau < 8: # fewer than 8 points
        # time series is too short for analysing using this fluctuation analysis. 
        logger.warning(f"This time series (N = {N}) is too short to analyze using this fluctuation analysis.")
        out = np.nan
    
    # 2) Compute the fluctuation function, F
    F = np.zeros(ntau)
    # each entry corresponds to a given scale, tau
    for i in range(ntau):
        # buffer the time series at the scale tau
        tau = taur[i]
        y_buff = _buffer(y, tau)
        if y_buff.shape[1] > int(np.floor(N/tau)): # zero-padded, remove trailing set of pts...
            y_buff = y_buff[:, :-1]

        # analyzed length of time series (with trailing pts removed)
        nn = y_buff.shape[1] * tau

        if wtf == 'nothing':
            y_dt = y_buff.reshape(nn, 1)
        elif wtf == 'endptdiff':
            # look at differences in end-points in each subsegment
            y_dt = y_buff[-1, :] - y_buff[0, :]
        elif wtf == 'range':
            y_dt = np.ptp(y_buff, axis=0)
        elif wtf == 'std':
            y_dt = np.std(y_buff, ddof=1, axis=0)
        elif wtf == 'iqr':
            y_dt = np.percentile(y_buff, 75, method='hazen', axis=0) - np.percentile(y_buff, 25, method='hazen', axis=0)
        elif wtf == 'dfa':
            tt = np.arange(1, tau + 1)[:, np.newaxis]
            for j in range(y_buff.shape[1]):
                p = np.polyfit(tt.ravel(), y_buff[:, j], k)
                y_buff[:, j] -= np.polyval(p, tt).ravel()
            y_dt = y_buff.reshape(-1)
        elif wtf == 'rsrange':
            # Remove straight line first: Caccia et al. Physica A, 1997
            # Straight line connects end points of each window:
            b = y_buff[0, :]
            m = y_buff[-1, :] - b
            y_buff -= (np.linspace(0, 1, tau)[:, np.newaxis] * m + b)
            y_dt = np.ptp(y_buff, axis=0)
        elif wtf == 'rsrangefit':
            # polynomial fit (order k) rather than endpoints fit: (~DFA)
            tt = np.arange(1, tau + 1)[:, np.newaxis]
            for j in range(y_buff.shape[1]):
                p = np.polyfit(tt.ravel(), y_buff[:, j], k)
                y_buff[:, j] -= np.polyval(p, tt).ravel()
            y_dt = np.ptp(y_buff, axis=0)
        else:
            raise ValueError(f"Unknwon fluctuation analysis method '{wtf}")
        
        F[i] = (np.mean(y_dt**q))**(1/q)

    # Smooth unevenly-distributed points in log space
    if logInc:
        logtt = np.log(taur)
        logFF = np.log(F)
        numTimeScales = ntau
    else:
        # need to smooth the unevenly-distributed pts (using a spline)
        logtaur = np.log(taur)
        logF = np.log(F)
        numTimeScales = 50 # number of sampling pts across the range
        logtt = np.linspace(np.min(logtaur), np.max(logtaur), numTimeScales) # even sampling in tau
        # equivalent to spline function in MATLAB
        spl_fit = CubicSpline(logtaur, logF)
        logFF = spl_fit(logtt)

    # Linear fit the log-log plot: full range
    out = {}
    out = doRobustLinearFit(out, logtt, logFF, range(numTimeScales), '')
    
    """ 
    WE NEED SOME SORT OF AUTOMATIC DETECTION OF GRADIENT CHANGES/NUMBER
    %% OF PIECEWISE LINEAR PIECES

    ------------------------------------------------------------------------------
    Try assuming two components (2 distinct scaling regimes)
    ------------------------------------------------------------------------------
    Move through, and fit a straight line to loglog before and after each point.
    Find point with the minimum sum of squared errors

    First spline interpolate to get an even sampling of the interval
    (currently, in the log scale, there are relatively more at slower timescales)

    Determine the errors
    """
    sserr = np.full(numTimeScales, np.nan) # don't choose the end pts
    minPoints = 6
    for i in range(minPoints, (numTimeScales-minPoints)+1):
        r1 = np.arange(i)
        p1 = np.polyfit(logtt[r1], logFF[r1], 1) # first degree polynomial
        r2 = np.arange(i-1, numTimeScales)
        p2 = np.polyfit(logtt[r2], logFF[r2], 1)
        sserr[i] = (np.linalg.norm(np.polyval(p1, logtt[r1]) - logFF[r1]) +
                    np.linalg.norm(np.polyval(p2, logtt[r2]) - logFF[r2]))
    
    # breakPt is the point where it's best to fit a line before and another line after
    breakPt = np.nanargmin(sserr)
    r1 = np.arange(breakPt)
    r2 = np.arange(breakPt, numTimeScales)

    # Proportion of the domain of timescales corresponding to the first good linear fit
    out['prop_r1'] = len(r1)/numTimeScales
    out['logtausplit'] = logtt[breakPt]
    out['ratsplitminerr'] = np.nanmin(sserr) / out['ssr']
    out['meanssr'] = np.nanmean(sserr)
    out['stdssr'] = np.nanstd(sserr, ddof=1)


    # Check that at least 3 points are available
    # Now we perform the robust linear fitting and get statistics on the two segments
    # R1
    out = doRobustLinearFit(out, logtt, logFF, r1, 'r1_')
    # R2
    out = doRobustLinearFit(out, logtt, logFF, r2, 'r2_')

    if np.isnan(out['r1_alpha']) or np.isnan(out['r2_alpha']):
        out['alpharat'] = np.nan
    else:
        out['alpharat'] = out['r1_alpha'] / out['r2_alpha']

    return out

def doRobustLinearFit(out, logtt, logFF, theRange, fieldName):
    """
    Get robust linear fit statistics on scaling range
    Adds fields to the output structure
    """
    if len(theRange) < 8 or np.all(np.isnan(logFF[theRange])):
        out[f'{fieldName}linfitint'] = np.nan
        out[f'{fieldName}alpha'] = np.nan
        out[f'{fieldName}se1'] = np.nan
        out[f'{fieldName}se2'] = np.nan
        out[f'{fieldName}ssr'] = np.nan
        out[f'{fieldName}resac1'] = np.nan
    else:
        X = sm.add_constant(logtt[theRange])
        model = sm.RLM(logFF[theRange], X, M=sm.robust.norms.TukeyBiweight())
        results = model.fit()
        out[f'{fieldName}linfitint'] = results.params[0]
        out[f'{fieldName}alpha'] = results.params[1]
        out[f'{fieldName}se1'] = results.bse[0]
        out[f'{fieldName}se2'] = results.bse[1]
        out[f'{fieldName}ssr'] = np.mean(results.resid ** 2)
        out[f'{fieldName}resac1'] = AutoCorr(results.resid, 1, 'Fourier')[0]
    
    return out


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

def FastDFA(x : list, intervals : Union[list, None] = None):
    """
    Perform detrended fluctuation analysis on a 
    nonstationary input signal.

    Adapted from the the original fastdfa code by Max A. Little and 
    publicly-available at http://www.maxlittle.net/software/index.php
    M. Little, P. McSharry, I. Moroz, S. Roberts (2006),
    Nonlinear, biophysically-informed speech pathology detection
    in Proceedings of ICASSP 2006, IEEE Publishers: Toulouse, France.

    Parameters:
    -----------
    x: 
        Input signal (must be a 1D numpy array)
    intervals: 
        Optional list of sample interval widths at each scale

    Returns:
    --------
    intervals: 
        List of sample interval widths at each scale
    flucts: 
        List of fluctuation amplitudes at each scale
    """
    if x.ndim != 1:
        raise ValueError("Input sequence must be a vector.")
    
    elements = len(x)
    
    if intervals is None:
        scales = int(np.log2(elements))
        if (1 << (scales - 1)) > elements / 2.5:
            scales -= 1
        intervals = _calculate_intervals(elements, scales)
    else:
        if len(intervals) < 2:
            raise ValueError("Number of intervals must be greater than one.")
        if np.any((intervals > elements) | (intervals < 3)):
            raise ValueError("Invalid interval size: must be between size of sequence x and 3.")
    
    y = np.cumsum(x) # get the cumualtive sum of the input time series
    # perform dfa to get back the flucts at each scale
    flucts = _dfa(y, intervals)
    # now fit a straight line to the log-log plot
    coeffs = np.polyfit(np.log10(intervals), np.log10(flucts), 1)
    alpha = coeffs[0]

    return alpha

# helper functions
def _calculate_intervals(elements, scales):
    # create an array of interval sizes using bitshifting to calculate powers of 2
    return np.array([int((elements / (1 << scale)) + 0.5) for scale in range(scales - 1, -1, -1)])

def _dfa(x, intervals):
    # measure the fluctuations at each scale
    elements = len(x)
    flucts = np.zeros(len(intervals))

    for scale, interval in enumerate(intervals):
        # calculate num subdivisions for this interval size
        subdivs = int(np.ceil(elements / interval))
        trend = np.zeros(elements)

        for i in range(subdivs):
            # calculate start and end indices for current subdivision
            start = i * interval
            end = start + interval
            # if last subdivision extends beyond end of the time series
            if end > elements:
                trend[start:] = x[start:]
                break
            segment = x[start:end]
            # extract the current segment of the detrended time series and fit a linear trend
            t = np.arange(interval)
            coeffs = np.polyfit(t, segment, 1)
            # store the trend values
            trend[start:end] = np.polyval(coeffs, t)
        # compute the root mean square fluctuations for the current interval size, after detrending
        flucts[scale] = np.sqrt(np.mean((x - trend)**2))

    return flucts
