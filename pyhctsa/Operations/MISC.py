#  functions which must be imported following others - provisional structure
import numpy as np
from typing import Union
from loguru import logger
from SY import SlidingWindow
from BF_iszscored import BF_iszscored
from IN import FirstCrossing, FirstMin
from scipy.interpolate import make_lsq_spline
from CO import AutoCorr
from scipy.stats import ansari, gaussian_kde
from statsmodels.sandbox.stats.runs import runstest_1samp
import pywt
from wrcoef import wavedec, wrcoef, detcoef

def WLCoeffs(y : list, wname : str = 'db3', level : Union[int, str] = 3):
    """
    Wavelet decomposition of the time series.

    Performs a wavelet decomposition of the time series using a given wavelet at a
    given level and returns a set of statistics on the coefficients obtained.

    Note: expect a discrepancy for the decay rate stats due to indexing differences between MATLAB and python.
    """

    N = len(y)
    if level == 'max':
        level = pywt.dwt_max_level(N, wname)
        if level == 0:
            raise ValueError("Cannot compute wavelet coefficients (short time series)")
    
    if pywt.dwt_max_level(N, wname) < level:
        raise ValueError(f"Chosen level, {level}, is too large for this wavelet on this signal.")
    
    C, L = wavedec(y, wavelet=wname, level=level)
    det = wrcoef(C, L, wname, level)
    det_s = np.sort(np.abs(det))[::-1]

    #%% Return statistics
    out = {}
    out['mean_coeff'] = np.mean(det_s)
    out['max_coeff'] = np.max(det_s)
    out['med_coeff'] = np.median(det_s)

    #% Decay rate stats ('where below _ maximum' = 'wb_m')
    out['wb99m'] = _findMyThreshold(0.99, det_s, N)
    out['wb90m'] = _findMyThreshold(0.90, det_s, N)
    out['wb75m'] = _findMyThreshold(0.75, det_s, N)
    out['wb50m'] = _findMyThreshold(0.50, det_s, N)
    out['wb25m'] = _findMyThreshold(0.25, det_s, N)
    out['wb10m'] = _findMyThreshold(0.10, det_s, N)
    out['wb1m'] = _findMyThreshold(0.01, det_s, N)

    return out

def _findMyThreshold(x, det_s, N):
    # helper function for WLCoeffs
    pr = np.argwhere(det_s < x*np.max(det_s))[0] / N
    if len(pr) == 0:
        return np.nan
    else:
        return pr[0]

def Walker(y : list, walkerRule : str = 'prop', walkerParams : Union[None, float, int, list] = None):
    """
    Simulates a hypothetical walker moving through the time domain.

    Note: due to differences in how the kde is implemented, exepct a discrepancy in the 
    `sw_distdiff` feature.
    """
    N = len(y)

    # Define default values and type requirements for each rule
    WALKER_CONFIGS = {
        'prop': {
            'default': 0.5,
            'valid_types': (int, float),
            'error_msg': 'must be float or integer'
        },
        'biasprop': {
            'default': [0.1, 0.2],
            'valid_types': (list,),
            'error_msg': 'must be a list'
        },
        'momentum': {
            'default': 2,
            'valid_types': (int, float),
            'error_msg': 'must be float or integer'
        },
        'runningvar': {
            'default': [1.5, 50],
            'valid_types': (list,),
            'error_msg': 'must be a list'
        }
    }

    if walkerRule not in WALKER_CONFIGS:
        valid_rules = ", ".join(f"'{rule}'" for rule in WALKER_CONFIGS.keys())
        raise ValueError(f"Unknown walker_rule: '{walkerRule}'. Choose from: {valid_rules}")
    
    # get configuration for the specified rule
    config = WALKER_CONFIGS[walkerRule]

    # use the default value if no parameters provided
    if walkerParams is None:
        walkerParams = config['default']

    if not isinstance(walkerParams, config["valid_types"]):
        raise ValueError(
            f"walkerParams {config['error_msg']} for walker rule: '{walkerRule}'"
        )
    
    # Do the walk
    w = np.zeros(N)

    if walkerRule == 'prop':
        #  % walker starts at zero and narrows the gap between its position
        #and the time series value at that point by the proportion given
        #in walkerParams, to give the value at the subsequent time step
        p = walkerParams
        w[0] = 0 # start at zero
        for i in range(1, N):
            w[i] = w[i-1] + p * (y[i-1] - w[i-1])
        
    elif walkerRule == 'biasprop':
        #walker is biased in one or the other direction (i.e., prefers to
        # go up, or down). Requires a vector of inputs: [p_up, p_down]
        pup, pdown = walkerParams

        w[0] = 0
        for i in range(1, N):
            if y[i] > y[i-1]: # time series inceases
                w[i] = w[i-1] + pup*(y[i-1]-w[i-1])
            else:
                w[i] = w[i-1] + pdown*(y[i-1]-w[i-1])
    elif walkerRule == 'momentum':
        #  % walker moves as if it had inertia from the previous time step,
        # i.e., it 'wants' to move the same amount; the time series acts as
        # a force changing its motion
        m = walkerParams # 'inertial mass'

        w[0] = y[0]
        w[1] = y[1]
        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1]-w[i-2])
            w[i] = w_inert + (y[i]-w_inert)/m # dissipative term
            #  % equation of motion (s-s_0=ut+F/m*t^2)
            # where the 'force' F is the change in the original time series
            # at that point
    elif walkerRule == 'runningvar':
        #  % walker moves with momentum defined by amplitude of past values in
        # a given length window
        m = walkerParams[0] # 'inertial mass'
        wl = int(walkerParams[1]) # window length

        w[0] = y[0]
        w[1] = y[1]
        for i in range(2, N):
            w_inert = w[i-1] + (w[i-1]-w[i-2])
            w_mom = w_inert + (y[i] - w_inert)/m # dissipative term from time series
            if i >= wl:
                w[i] = w_mom*(np.std(y[i-wl:(i+1)], ddof=1)/np.std(w[i-wl:(i+1)], ddof=1)) # adjust by local standard deviation
            else:
                w[i] = w_mom
    
    # Get statistics on the walk
    out = {}
    # the walk itself
    out['w_mean'] = np.mean(w)
    out['w_median'] = np.median(w)
    out['w_std'] = np.std(w, ddof=1)
    out['w_ac1'] = AutoCorr(w, 1, 'Fourier')[0] # lag 1 autocorr
    out['w_ac2'] = AutoCorr(w, 2, 'Fourier')[0] # lag 2 autocorr
    out['w_tau'] = FirstCrossing(w, 'ac', 0, 'continuous')
    out['w_min'] = np.min(w)
    out['w_max'] = np.max(w)
    # fraction of time series length that walker crosses time series
    out['w_propzcross'] = (np.sum((w[:-1] * w[1:]) < 0)) / (N-1)

    # Differences between the walk at signal
    out['sw_meanabsdiff'] = np.mean(np.abs(y - w))
    out['sw_taudiff'] = FirstCrossing(y, 'ac', 0, 'continuous') - FirstCrossing(w, 'ac', 0 , 'continuous')
    out['sw_stdrat'] =  np.std(w, ddof=1)/np.std(y, ddof=1)
    out['sw_ac1rat'] = out['w_ac1']/AutoCorr(y, 1)[0]
    out['sw_minrat'] = np.min(w)/np.min(y)
    out['sw_maxrat'] = np.max(w)/np.max(y)
    out['sw_propcross'] = np.sum((w[:-1] - y[:-1]) * (w[1:] - y[1:]) < 0)/(N-1)

    #% test from same distribution: Ansari-Bradley test
    _, pval = ansari(w, y)
    out['sw_ansarib_pval'] = pval

    r = np.linspace(
        min(min(y), min(w)),
        max(max(y), max(w)),
        200
    )

    kde_y = gaussian_kde(y)
    kde_w = gaussian_kde(w)
    dy = kde_y(r)
    dw = kde_w(r)
    out['sw_distdiff'] = np.sum(np.abs(dy - dw))

    #% (iii) Looking at residuals between time series and walker
    res = w - y
    _, runs_pval = runstest_1samp(res, cutoff='mean')
    out['res_runstest'] = runs_pval
    out['res_swss5_1'] = SlidingWindow(res, 'std', 'std', 5, 1) # sliding window stationarity
    out['res_ac1'] = AutoCorr(res, 1)[0] # auto correlation at lag-1

    return out


def ForcePotential(y : list, whatPotential : str = 'dblwell', params : Union[list, None] = None) -> dict:
    """
    Couples the values of the time series to a dynamical system.

    The input time series forces a particle in the given potential well.

    Args:
    y (array-like): The input time series.
    what_potential (str): The potential function to simulate:
                          'dblwell' (a double well potential function) or
                          'sine' (a sinusoidal potential function).
    params (list): The parameters for simulation, should be in the form:
                   [alpha, kappa, deltat]

    Returns:
    dict: Statistics summarizing the trajectory of the simulated particle.
    """
    if params is None:
        if whatPotential == 'dblwell':
            params = [2, 0.1, 0.1]
        elif whatPotential == 'sine':
            params = [1, 1, 1]
        else:
            ValueError(f"Unknown system {whatPotential}")
    else:
        # check params
        if not isinstance(params, list):
            raise ValueError("Expected list of parameters.")
        else:
            if len(params) != 3:
                raise ValueError("Expected 3 parameters.")
    
    N = len(y) # length of the time series

    alpha, kappa, deltat = params

    # specify potential function
    if whatPotential == 'sine':
        V = lambda x: -np.cos(x/alpha)
        F = lambda x: np.sin(x/alpha)/alpha
    elif whatPotential == 'dblwell':
        F = lambda x: -x**3 + alpha**2 * x
        V = lambda x: x**4 / 4 - alpha**2 * x**2 / 2
    else:
        raise ValueError(f"Unknown potential function {whatPotential}")
    
    x = np.zeros(N) # position
    v = np.zeros(N) # velocity

    for i in range(1, N):
        x[i] = x[i-1] + v[i-1]*deltat + (F(x[i-1]) + y[i-1] - kappa*v[i-1])*deltat**2
        v[i] = v[i-1] + (F(x[i-1]) + y[i-1] - kappa*v[i-1])*deltat

    # check the trajectory didn't blow out
    if np.isnan(x[-1]) or np.abs(x[-1]) > 1E10:
        return np.NaN
    
    # Output some basic features of the trajectory
    out = {}
    out['mean'] = np.mean(x) # mean position
    out['median'] = np.median(x) # median position
    out['std'] = np.std(x, ddof=1) # std. dev.
    out['range'] = np.ptp(x)
    out['proppos'] = np.sum(x >0)/N
    out['pcross'] = np.sum(x[:-1] * x[1:] < 0) / (N - 1)
    out['ac1'] = np.abs(AutoCorr(x, 1, 'Fourier')[0])
    out['ac10'] = np.abs(AutoCorr(x, 10, 'Fourier')[0])
    out['ac50'] = np.abs(AutoCorr(x, 50, 'Fourier')[0])
    out['tau'] = FirstCrossing(x, 'ac', 0, 'continuous')
    out['finaldev'] = np.abs(x[-1]) # final position

    # additional outputs for dbl well
    if whatPotential == 'dblwell':
        out['pcrossup'] = np.sum((x[:-1] - alpha) * (x[1:] - alpha) < 0) / (N - 1)
        out['pcrossdown'] = np.sum((x[:-1] + alpha) * (x[1:] + alpha) < 0) / (N - 1)

    return out

def PeriodicityWang(y : list) -> np.ndarray:
    # note: due to indexing differences, each python output will be one less than MATLAB output
    # check time series is z-scored
    if not BF_iszscored(y):
        logger.warning("The input time series should be z-scored.")
    N = len(y)
    xdata = np.arange(0, N)
    ths = [0, 0.01,0.1,0.2,1/np.sqrt(N),5/np.sqrt(N),10/np.sqrt(N)]
    numThresholds = len(ths)
    # detrend using a regression spline with 3 knots
    numPolyPieces = 2 # number of polynomial pieces in the spline
    breaks = np.linspace(xdata[0], xdata[-1], numPolyPieces)
    splineOrder = 4 # order of the spline
    t = np.concatenate((
        np.full(splineOrder+1, breaks[0]),
        breaks[0:-1],
        np.full(splineOrder+1, breaks[-1])
    ))
    spline = make_lsq_spline(x=xdata, y=y, t=t, k=splineOrder)
    y_spl = spline(np.arange(0, N))
    y = y - y_spl

    # 2. Compute autocorrelations up to 1/3 the length of the time series.
    acmax = int(np.ceil(N/3)) # compute the autocorrelation up to this lag
    acf = np.zeros(acmax)
    for i in range(0, acmax):
        acf[i] = np.mean(y[:N-i-1] * y[i+1:N+1])
    
    # 3. Frequency is the first peak satisfying the following conditions:
    diffac = np.diff(acf) # % differenced time series
    sgndiffac = np.sign(diffac)
    bath = np.diff(sgndiffac)
    troughs = np.argwhere(bath == 2).flatten() + 1 # % finds troughs
    peaks = np.argwhere(bath == -2).flatten() + 1 # % finds peaks
    numPeaks = len(peaks)

    theFreqs = np.zeros(numThresholds)
    for k in range(numThresholds):
        theFreqs[k] = 1
        for i in range(numPeaks):
            ipeak = peaks[i] # index
            thepeak = acf[ipeak] # get the acf at the peak
            ftrough = np.argwhere(troughs < ipeak).flatten()[-1]
            if ftrough.size == 0:
                continue  # no trough found before ipeak
            itrough = troughs[ftrough]
            theTrough = acf[itrough]
    
            if thepeak - theTrough < ths[k]:
                continue

            if thepeak < 0:
                continue
            
            theFreqs[k] = ipeak
            break 

    return theFreqs

def TRev(y : list, tau : Union[int, str, None] = 'ac'):
    """
    Normalized nonlinear autocorrelation, trev function of a time series.

    Calculates the trev function, a normalized nonlinear autocorrelation,
    mentioned in the documentation of the TSTOOL nonlinear time-series analysis
    package.

    Parameters:
    y (array-like): Time series
    tau (int, str, optional): Time lag. Can be 'ac' or 'mi' to set as the first 
                              zero-crossing of the autocorrelation function, or 
                              the first minimum of the automutual information 
                              function, respectively. Default is 'ac'.

    Returns:
    dict: A dictionary containing the following keys:
        - 'raw': The raw trev expression
        - 'abs': The magnitude of the raw expression
        - 'num': The numerator
        - 'absnum': The magnitude of the numerator
        - 'denom': The denominator

    Raises:
    ValueError: If no valid setting for time delay is found.
    """

    # Can set the time lag, tau, to be 'ac' or 'mi'
    if tau == 'ac':
        # tau is first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = FirstMin(y, 'mi')
    if np.isnan(tau):
        raise ValueError("No valid setting for time delay. (Is the time series too short?)")

    # Compute trev quantities
    yn = y[:-tau]
    yn1 = y[tau:] # yn, tau steps ahead
    
    out = {}

    # The trev expression used in TSTOOL
    raw = np.mean((yn1 - yn)**3) / (np.mean((yn1 - yn)**2))**(3/2)
    out['raw'] = raw

    # The magnitude
    out['abs'] = np.abs(raw)

    # The numerator
    num = np.mean((yn1-yn)**3)
    out['num'] = num
    out['absnum'] = np.abs(num)

    # the denominator
    out['denom'] = (np.mean((yn1-yn)**2))**(3/2)

    return out

def TC3(y : list, tau : Union[int, str, None] = 'ac'):
    """
    Normalized nonlinear autocorrelation function, tc3.

    Computes the tc3 function, a normalized nonlinear autocorrelation, at a
    given time-delay, tau.
    Statistic is for two time-delays, normalized in terms of a single time delay.
    Used as a test statistic for higher order correlational moments in surrogate
    data analysis.

    Parameters:
    y (array-like): Input time series
    tau (int or str, optional): Time lag. If 'ac' or 'mi', it will be computed.

    Returns:
    dict: A dictionary containing:
        - 'raw': The raw tc3 expression
        - 'abs': The magnitude of the raw expression
        - 'num': The numerator
        - 'absnum': The magnitude of the numerator
        - 'denom': The denominator

    Note: This function requires the implementation of CO_FirstCrossing and 
    CO_FirstMin functions, which are not provided in this conversion.
    """

    # Set the time lag as a measure of the time-series correlation length
    # Can set the time lag, tau, to be 'ac' or 'mi'
    if tau == 'ac':
        # tau is first zero crossing of the autocorrelation function
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        # tau is the first minimum of the automutual information function
        tau = FirstMin(y, 'mi')
    
    if np.isnan(tau):
        raise ValueError("No valid setting for time delay (time series too short?)")
    
    # Compute tc3 statistic
    yn = y[:-2*tau]
    yn1 = y[tau:-tau] # yn1, tau steps ahead
    yn2 = y[2*tau:] # yn2, 2*tau steps ahead

    numerator = np.mean(yn * yn1 * yn2)
    denominator = np.abs(np.mean(yn * yn1)) ** (3/2)

    # The expression used in TSTOOL tc3:
    out = {}
    out['raw'] = numerator / denominator

    # The magnitude
    out['abs'] = np.abs(out['raw'])

    # The numerator
    out['num'] = numerator
    out['absnum'] = np.abs(out['num'])

    # The denominator
    out['denom'] = denominator

    return out

def TranslateShape(y : list, shape : str = 'circle', d : Union[int, float] = 2, howToMove : str = 'pts'):
    """
    Statistics on datapoints inside geometric shapes across the time series.

    """
    y = np.array(y, dtype=float)
    N = len(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.shape[1] > y.shape[0]:
        y = y.T

    # add a time index
    ty = np.column_stack((np.arange(1, N+1), y[:, 0])) # has increasing integers as time in the first column
    #-------------------------------------------------------------------------------
    # Generate the statistics on the number of points inside the shape as it is
    # translated across the time series
    #-------------------------------------------------------------------------------
    if howToMove == 'pts':

        if shape == 'circle':

            r = d # set radius
            w = int(np.floor(r))
            rnge = np.arange(1 + w, N - w + 1)
            NN = len(rnge) # number of admissible points
            np_counts = np.zeros(NN, dtype=int)

            for i in range(NN):
                idx = rnge[i]
                start = idx - w - 1
                end = idx + w
                win = ty[start:end, :]
                difwin = win - ty[idx - 1, :]
                squared_dists = np.sum(difwin**2, axis=1)
                np_counts[i] = np.sum(squared_dists <= r**2)

        elif shape == 'rectangle':

            w = d
            rnge = np.arange(1 + w, N - w + 1)
            NN = len(rnge)
            np_counts = np.zeros(NN, dtype=int)

            for i in range(NN):
                idx = rnge[i]
                start = (idx - w) - 1
                end = (idx + w)
                np_counts[i] = np.sum(
                    np.abs(y[start:end, 0]) <= np.abs(y[i, 0])
                )
        else:
            raise ValueError(f"Unknown shape {shape}. Choose either 'circle' or 'rectangle'")
    else:
        raise ValueError(f"Unknown setting for 'howToMove' input: '{howToMove}'. Only option is currently 'pts'.")

    # compute stats on number of hits inside the shape
    out = {}
    out["max"] = np.max(np_counts)
    out["std"] = np.std(np_counts, ddof=1)
    out["mean"] = np.mean(np_counts)
    
    # count the hits
    vals, hits = np.unique_counts(np_counts)
    max_val = np.argmax(hits)
    out["npatmode"] = hits[max_val]/NN
    out["mode"] = vals[max_val]

    count_types = ["ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens", "elevens"]
    for i in range(1, 12):
        if 2*w + 1 >= i:
            out[f"{count_types[i-1]}"] = np.mean(np_counts == i)
    
    out['statav2_m'] = SlidingWindow(np_counts, 'mean', 'std', 2, 1)
    out['statav2_s'] = SlidingWindow(np_counts, 'std', 'std', 2, 1)
    out['statav3_m'] = SlidingWindow(np_counts, 'mean', 'std', 3, 1)
    out['statav3_s'] = SlidingWindow(np_counts, 'std', 'std', 3, 1)
    out['statav4_m'] = SlidingWindow(np_counts, 'mean', 'std', 4, 1)
    out['statav4_s'] = SlidingWindow(np_counts, 'std', 'std', 4, 1)

    return out
