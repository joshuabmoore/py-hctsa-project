import antropy as ant
from antropy.utils import _embed, _xlogx
from math import factorial
import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing

def _perm_entropy_all(x, order=3, delay=1, normalize=False, return_normedCounts=False):
    # compute all relevant perm entropy stats
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean([_perm_entropy_all(x, order=order, delay=d, normalize=normalize) for d in delay])
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    assert delay > 0, "delay must be greater than zero."
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -_xlogx(p).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    
    if return_normedCounts:
        return pe, p
    else:
        return pe

def EN_PermEn(y, m = 2, tau = 1):
    """
    Permutation Entropy of a time series.

    "Permutation Entropy: A Natural Complexity Measure for Time Series"
    C. Bandt and B. Pompe, Phys. Rev. Lett. 88(17) 174102 (2002)

    Parameters:
    -----------
    y : array-like
        the input time series
    m : integer
        the embedding dimension (or order of the permutation entropy)
    tau : int or str
        the time-delay for the embedding
    Returns:
    out : dict
        Outputs the permutation entropy and normalized version computed according to
        different implementations
    --------
    """
    if tau == 'ac':
        tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
    elif not isinstance(tau, int):
        raise TypeError("Invalid type for tau. Can be either 'ac' or an integer.")
    
    pe, p = _perm_entropy_all(y, order=m, delay=tau, normalize=False, return_normedCounts=True)
    pe_n = ant.perm_entropy(y, order=m, delay=tau, normalize=True)
    Nx = len(y) - (m-1) * tau # get the number of embedding vectors
    # p will only contain non-zero probabilities, so to make the output consistent with MATLAB, we need to add a correction:
    # not saying this is correct, but this is how it is implemented in MATLAB and this is a port...
    lenP = len(p)
    numZeros = factorial(m) - lenP
    # append the zeros to the end of p
    p = np.concatenate([np.array(p), np.zeros(numZeros)])
    p_LE = [np.maximum(1/Nx, p[i]) for i in range(len(p))]
    permEnLE = -np.sum(p_LE * np.log(p_LE))/(m-1)

    out = {}
    out['permEn'] = pe
    out['normPermEn'] = pe_n
    out['permEnLE'] = permEnLE

    return out
