import numpy as np
from Operations.CO_AutoCorr import CO_AutoCorr
from warnings import warn
from scipy.interpolate import CubicSpline
import statsmodels.api as sm

def SC_FluctAnal(x, q = 2, wtf = 'rsrange', tauStep = 1, k = 1, lag = None, logInc = True):
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
        warn(f"This time series (N = {N}) is too short to analyze using this fluctuation analysis.")
        out = np.NaN
    
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
        out[f'{fieldName}resac1'] = CO_AutoCorr(results.resid, 1, 'Fourier')[0]
    
    return out
