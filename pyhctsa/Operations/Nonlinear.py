import numpy as np 
from pyhctsa.Toolboxes.Michael_Small.ms_nearest_wrapper import ms_nearest
from numpy.typing import ArrayLike
from typing import Iterable, Tuple, Union, Optional, List
from numba import njit, prange
from Correlation import FirstCrossing
from Information import FirstMin


def FNN(y: Union[list, np.ndarray], 
        de: Union[List[int], None] = list(np.arange(1, 11)), 
        tau: Union[int, str] = 1,
        th: float = 5.0, 
        kth: int = 1, 
        justBest: bool = False, 
        bestp: float = 0.1):
    """
    False nearest neighbors of a time series.

    Determines the number of false nearest neighbors for the embedded time series
    using Michael Small's false nearest neighbor code, fnn (renamed MS_fnn here).

    False nearest neighbors are judged using a ratio of the distances between the
    next k points and the neighboring points of a given datapoint.

    Parameters
    ----------
    y : list or np.ndarray
        Input time series (1D).

    de : int, list of int, or np.ndarray, optional
        Embedding dimensions to compare across (a vector). Default: list(range(1, 11))

    tau : int or str, optional
        The time-delay. Can be:
            - int: a fixed lag
            - 'ac': first zero-crossing of the autocorrelation function (ACF)
            - 'mi': first minimum of the auto mutual information (AMI)
        Default: 1

    th : float, optional
        The distance threshold for neighbours. Default: 5.0

    kth : int, optional
        The distance to next points (step ahead for neighbour test). Default: 1

    justBest : bool, optional
        If True, just return the best embedding dimension, m_best. Default: True

    bestp : float, optional
        If justBest is True, bestp is the proportion of false nearest
        neighbours at which the optimal embedding dimension is selected. Default: 0.1

    Returns
    -------
    If justBest:
        int : Best embedding dimension (m_best).
    Else:
        dict : Statistics on the proportion of false nearest neighbors as a function
               of the embedding dimension m = m_min, m_min+1, ..., m_max for a given
               time lag tau and distance threshold for neighbors, d_th. Includes:
                   - Proportion of false nearest neighbors at each m
                   - The mean and spread
                   - The smallest m at which the proportion drops below each of a set
                     of fixed thresholds

    References
    ----------
    M. Small, Applied Nonlinear Time Series Analysis: Applications in Physics,
    Physiology, and Finance. World Scientific, Nonlinear Science Series A,
    Vol. 52 (2005).
    Code available at http://small.eie.polyu.edu.hk/matlab/
    """
    y = np.asarray(y)
    
    if tau == 'ac':
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
    elif tau == 'mi':
        tau = FirstMin(y, 'mi')
    elif not isinstance(tau, int) or tau < 1:
        raise ValueError("tau must be a positive integer or 'ac'/'mi'.")
    
    if np.isnan(tau):
        raise ValueError("Time series too short for FNN.")
    
    p = _fnn_props(y, de, tau, th, kth).flatten()
    firstunderf = lambda p, de, x: de[np.where(p < x)[0][0]] if np.any(p < x) else de[-1] + 1

    if justBest:
        return firstunderf(p, de, bestp)
    else:
        out = {f'pfnn_{de[i]}': p[i] for i in range(len(de))}
        out['meanpfnn'] = np.mean(p)
        out['stdpfnn'] = np.std(p, ddof=1)
        out['firstunder02'] = firstunderf(p, de, 0.2)
        out['firstunder01'] = firstunderf(p, de, 0.1)
        out['firstunder005'] = firstunderf(p, de, 0.05)
        out['firstunder002'] = firstunderf(p, de, 0.02)
        out['firstunder001'] = firstunderf(p, de, 0.01)
        out['max1stepchange'] = np.max(np.abs(np.diff(p)))
        return out

def _fnn_props(y, de=None, tau=None, th=5, kth=1):
    """
    Determines the proportion of false nearest neighbours for the time
    series y embedded in dimension de with lag tau.

    Parameters
    ----------
    y : array-like
        Input time series.
    de : int or array-like, optional
        Embedding dimensions to test. Default: np.arange(1, 11)
    tau : int or array-like, optional
        Embedding lag(s) to test. Default: 1
    th : float, optional
        Threshold for false neighbour ratio. Default: 5
    kth : int, optional
        Step ahead for neighbour test. Default: 1

    Returns
    -------
    p : np.ndarray
        Proportion of false nearest neighbours for each (de, tau) pair.
        Shape: (len(de), len(tau))
    """
    y = np.asarray(y).ravel()
    if de is None:
        de = np.arange(1, 11)
    else:
        de = np.atleast_1d(de)
    if tau is None:
        tau = np.array([1])
    else:
        tau = np.atleast_1d(tau)

    p = []
    for t in tau:
        px = []
        for d in de:
            # Embed the data
            X = _ms_embed(y, d, t)
            dx, nx = X.shape
            X = np.asfortranarray(X)
            # Find the nearest neighbours of each point (exclude last kth columns)
            ind = _nearest_native(X[:, :nx-kth], np.ones(dx), t)
            #ind = ms_nearest(X[:, :nx-kth], t, np.ones(dx))

            # Distance between each point and its nearest neighbour
            d0 = _ms_rms((X[:, :nx - kth].T - X[:, ind].T))
            # ... and after one time step
            d1 = _ms_rms((X[:, kth:nx].T - X[:, ind + 1].T))

            # Exclude any coincident points
            mask = d0.flatten() != 0
            d0 = d0[mask]
            d1 = d1[mask]

            # Calculate the proportion of false nearest neighbours
            if d0.size == 0:
                ifnn = 0.0
            else:
                ifnn = np.sum(np.divide(d1, d0) > th) / len(d0)

            px.append(ifnn)
        p.append(px)
    p = np.array(p).T 
    return p

@njit(parallel=True, fastmath=True)
def _nearest_native(x: np.ndarray,
            avect: np.ndarray,
            tau: int) -> np.ndarray:
    """
    Find, for every column of `x`, the index of the nearest-neighbour column
    (excluding those within the Theiler window `tau`).

    Parameters
    ----------
    x : (m, n) float64 ndarray
        Matrix whose *columns* are m-dimensional delay-vectors.
        **Shape must be (m, n)** – i.e. the same orientation MATLAB uses.
    avect : (m,) float64 ndarray
        Weighting vector applied element-wise to the squared differences.
        (Use `np.ones(m)` for unweighted Euclidean distance.)
    tau : int
        Theiler window; columns with |i − j| ≤ tau are ignored.

    Returns
    -------
    ind : (n,) int64 ndarray
        Zero-based indices of each column’s nearest neighbour.

    Notes
    -----
    • Complexity O(n² m); compiled with Numba + `parallel=True` so every
      *outer* loop iteration can run in parallel threads.
    • `fastmath=True` enables FMA and other aggressive FP optimisations.
    """
    m, n = x.shape
    ind = np.empty(n, dtype=np.int64)

    # Pre-compute x² * avect so we can use a fast dot later if desired
    for i in prange(n):                # outer loop parallelised
        bestdist = np.inf
        closest  = -1

        for j in range(n):
            if abs(i - j) <= tau:      # Theiler window skip
                continue

            # Weighted squared Euclidean distance
            dist = 0.0
            for k in range(m):
                diff = x[k, i] - x[k, j]
                dist += diff * diff * avect[k]

            if dist < bestdist:        # keep running best
                bestdist = dist
                closest  = j

        ind[i] = closest

    return ind

def _ms_rms(y: np.ndarray) -> np.ndarray:
    """Row-wise RMS - returns a 1-D array."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return np.abs(y)
    return np.sqrt(np.mean(y**2, axis=-1))  # no reshape
        
def _ms_embed(z, *args, split=False):
    """
    Port of MS_embed.m (Michael Small, 2005) to Python / NumPy.

    Parameters
    ----------
    z : array_like
        One-dimensional signal to embed.
    *args :
        • ms_embed(z)                         → default lags [0, 1, 2]  
        • ms_embed(z, lags)                   → explicit 1-D iterable of lags  
        • ms_embed(z, dim, lag)               → shorthand: lags = 0, lag, …, lag*(dim-1)
    split : bool, default = False
        If True, return a tuple ``(x, y)`` where ``x`` contains all non-negative
        lags (rows) and ``y`` contains the negative lags.  If False, return a
        single array that stacks every requested lag in ascending order.

    Returns
    -------
    x : ndarray
        Embedded matrix of shape (n_lags_nonneg, n_windows) when *split=True*,
        or (len(lags), n_windows) when *split=False*.
    y : ndarray (only if *split=True*)
        Rows corresponding to negative lags.  Empty if no negatives were
        requested.

    Notes
    -----
    • Negative lags look **forward** in the series, positive lags look **back**.  
    • Columns (time) are arranged so the earliest usable sample is column 0.  
    • Raises ``ValueError`` if `z` is not 1-D or is too short for the
      requested lag window.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.arange(10)

    Default three-dimensional embedding (lags = [0, 1, 2]):
    >>> ms_embed(z)
    array([[2., 3., 4., 5., 6., 7., 8., 9.],
           [1., 2., 3., 4., 5., 6., 7., 8.],
           [0., 1., 2., 3., 4., 5., 6., 7.]])

    Explicit (dim = 3, lag = 2) → rows: t, t-2, t-4:
    >>> ms_embed(z, 3, 2)
    array([[4., 5., 6., 7., 8., 9.],
           [2., 3., 4., 5., 6., 7.],
           [0., 1., 2., 3., 4., 5.]])

    Split negative and non-negative lags:
    >>> x_pos, x_neg = ms_embed(z, [-1, 0, 1, 2], split=True)
    """
    # ------------------------------------------------------------------
    # Resolve the lags argument(s)
    if len(args) == 2:                         # (dim, lag)
        dim, lag = args
        lags = np.arange(dim) * lag
    elif len(args) == 1:                       # (lags,)
        lags = np.asarray(args[0], dtype=int)
    elif len(args) == 0:                       # no extra args
        lags = np.array([0, 1, 2], dtype=int)
    else:
        raise TypeError(
            "ms_embed expects ms_embed(z [, lags] "
            "or ms_embed(z, dim, lag))"
        )

    lags = np.asarray(lags, dtype=int)
    lags.sort()

    # ------------------------------------------------------------------
    # Input checks
    z = np.asarray(z, dtype=float).squeeze()
    if z.ndim != 1:
        raise ValueError("`z` must be a 1-D array (vector).")

    n = z.size
    max_lag = lags[-1]
    if n < max_lag:
        raise ValueError("Input vector is too short for the requested lags.")

    # ------------------------------------------------------------------
    # Build the embedded matrix
    window = max_lag - lags[0]           # total lag window
    m = n - window                       # number of embedding columns
    t = np.arange(m) + max_lag           # centre times (0-based)

    x_all = np.empty((lags.size, m), dtype=z.dtype)
    for i, lag in enumerate(lags):
        x_all[i] = z[t - lag]

    # ------------------------------------------------------------------
    # Return form
    if split:
        neg_mask = lags < 0
        y = x_all[neg_mask]
        x = x_all[~neg_mask]
        return (x, y) if y.size else (x, np.empty((0, m), dtype=z.dtype))
    else:
        return x_all
    