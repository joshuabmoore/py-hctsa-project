import scipy
import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing
from Operations.SB_CoarseGrain import SB_CoarseGrain

def SB_TransitionMatrix(y, howtocg = 'quantile', numGroups = 2, tau = 1):
    """
    Transition probabilities between time-series states. 
    The time series is coarse-grained according to a given method.

    The input time series is transformed into a symbolic string using an
    equiprobable alphabet of numGroups letters. The transition probabilities are
    calculated at a lag tau.

    Related to the idea of quantile graphs from time series.
    cf. Andriana et al. (2011). Duality between Time Series and Networks. PLoS ONE.
    https://doi.org/10.1371/journal.pone.0023378

    Parameters:
    -----------
    y : array_like
        Input time series (column vector)
    howtocg : str, optional
        The method of discretization (currently 'quantile' is the only
        option; could incorporate SB_CoarseGrain for more options in future)
    numGroups : int, optional
        number of groups in the course-graining
    tau : int or str, optional
        analyze transition matricies corresponding to this lag. We
        could either downsample the time series at this lag and then do the
        discretization as normal, or do the discretization and then just
        look at this dicrete lag. Here we do the former. Can also set tau to 'ac'
        to set tau to the first zero-crossing of the autocorrelation function.

    Returns:
    --------
    out : dict 
        A dictionary including the transition probabilities themselves, as well as the trace
        of the transition matrix, measures of asymmetry, and eigenvalues of the
        transition matrix.
    """
    # check inputs
    if numGroups < 2:
        raise ValueError("Too few groups for coarse-graining")
    if tau == 'ac':
        # determine the tau from first zero of the ACF
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
        if np.isnan(tau):
            raise ValueError("Time series too short to estimate tau")
    if tau > 1: # calculate transition matrix at a non-unit lag
        # downsample at rate 1:tau
        y = scipy.signal.resample(y, int(np.ceil(len(y) / tau)))
    
    N = len(y)

    # ------------------------------------------------------------------------------
    # (((1))) Discretize the time series to a symbolic string
    # ------------------------------------------------------------------------------

    yth = SB_CoarseGrain(y, howtocg, numGroups)
    # At this point we should have:
    # (*) yth: a thresholded y containing integers from 1 to numGroups
    yth = np.ravel(yth)
    
    # ------------------------------------------------------------------------------
    # (((2))) Compute the tau-step transition matrix
    #               (Markov for tau = 1)
    # ------------------------------------------------------------------------------

    T = np.zeros((numGroups,numGroups))
    for i in range(numGroups):
        ri = (yth == i + 1)
        if sum(ri) == 0:
            T[i,:] = 0
        else:
            ri_next = np.r_[False, ri[:-1]]
            for j in range(numGroups):
                T[i, j] = np.sum(yth[ri_next] == j + 1)

    out = {}
    # Normalize from counts to probabilities:
    T = T/(N - 1) # N-1 is appropriate because it's a 1-time transition matrix

    # ------------------------------------------------------------------------------
    # (((3))) Output measures from the transition matrix
    # ------------------------------------------------------------------------------
    # (i) Raw values of the transition matrix
    # [this has to be done bulkily (only for numGroups = 2,3)]

    if numGroups == 2:
        for i in range(4):
            out[f'T{i+1}'] = T.transpose().flatten()[i] # transpose to match MATLAB column major

    elif numGroups == 3:
        for i in range(9):
            out[f'T{i+1}'] = T.transpose().flatten()[i] # transpose to match MATLAB column major

    elif numGroups > 3:
        for i in range(numGroups):
            out[f'TD{i+1}'] = T.transpose()[i, i]

    # (ii) Measures on the diagonal
    out['ondiag'] = np.trace(T) # trace
    out['stddiag'] = np.std(np.diag(T), ddof=1) # std of diagonal elements

    # (iii) Measures of symmetry:
    out['symdiff'] = np.sum(np.abs(T - T.T)) # sum of differences of individual elements
    out['symsumdiff'] = np.sum(np.tril(T, -1)) - np.sum(np.triu(T, 1)) # difference in sums of upper and lower triangular parts of T

    # Measures from eigenvalues of T
    eig_T = np.linalg.eigvals(T)
    out['stdeig'] = np.std(eig_T, ddof=1)
    out['maxeig'] = np.max(np.real(eig_T))
    out['mineig'] = np.min(np.real(eig_T))
    out['maximeig'] = np.max(np.imag(eig_T))

    # Measures from covariance matrix
    cov_T = np.cov(T.transpose()) # need to transpose T to get same output as MATLAB's cov func. 
    out['sumdiagcov'] = np.trace(cov_T)

    # Eigenvalues of covariance matrix
    eig_cov_T = np.linalg.eigvals(cov_T)
    out['stdeigcov'] = np.std(eig_cov_T, ddof=1)
    out['maxeigcov'] = np.max(eig_cov_T)
    out['mineigcov'] = np.min(eig_cov_T)

    return out
