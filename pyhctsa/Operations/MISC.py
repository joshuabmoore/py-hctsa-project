#  functions which must be imported following others - provisional structure
import numpy as np
from typing import Union
from loguru import logger
from SY_SlidingWindow import SY_SlidingWindow
from BF_iszscored import BF_iszscored
from IN import FirstCrossing, FirstMin
from scipy.interpolate import make_lsq_spline
from CO import AutoCorr

def Walker(y : list, walkerRule : str ):
    """
    Simulates a hypothetical walker moving through the time domain.
    """

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
    
    out['statav2_m'] = SY_SlidingWindow(np_counts, 'mean', 'std', 2, 1)
    out['statav2_s'] = SY_SlidingWindow(np_counts, 'std', 'std', 2, 1)
    out['statav3_m'] = SY_SlidingWindow(np_counts, 'mean', 'std', 3, 1)
    out['statav3_s'] = SY_SlidingWindow(np_counts, 'std', 'std', 3, 1)
    out['statav4_m'] = SY_SlidingWindow(np_counts, 'mean', 'std', 4, 1)
    out['statav4_s'] = SY_SlidingWindow(np_counts, 'std', 'std', 4, 1)

    return out
