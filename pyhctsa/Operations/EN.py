# entropy module 
import numpy as np
import antropy as ant
from antropy.utils import _embed, _xlogx
from BF_zscore import BF_zscore
from BF_makeBuffer import BF_MakeBuffer
from BF_PreProcess import BF_PreProcess
from PN_sampenc import PN_sampenc
from CO import FirstCrossing
from typing import Union
from math import factorial

def WEntropy(y : list, whaten : str = 'shannon', p : Union[any, None] = None):
    """
    Entropy of time series using wavelets.
    Uses a python port of the MATLAB wavelet toolbox wentropy function.

    Parameters:
    ----------
    y : array_like
        Input time series
    whaten : str, optional
        The entropy type:
        - 'shannon' (default)
        - 'logenergy'
        - 'threshold' (with a given threshold)
        - 'sure' (with a given parameter)
        (see the wentropy documentation for information)
    p : any, optional
        the additional parameter needed for threshold and sure entropies

    Returns:
    --------
    out : float
        Entropy value. 
    """
    N = len(y)

    if whaten == 'shannon':
        # compute Shannon entropy
        out = _wentropy(y, 'shannon')/N
    elif whaten == 'logenergy':
        out = _wentropy(y, 'logenergy')/N
    elif whaten == 'threshold':
        # check that p has been provided
        if p is not None:
            out = _wentropy(y, 'threshold', p)/N
        else:
            raise ValueError("threshold requires an additional parameter, p.")
    elif whaten == 'sure':
        if p is not None:
            out = _wentropy(y, 'sure', p)/N
        else:
            raise ValueError("sure requires an additional parameter, p.")
    else:
        raise ValueError(f"Unknown entropy type {whaten}")

    return out

def _wentropy(x, entType = 'shannon', additionalParameter = None):
    # helper function
    # taken from https://github.com/fairscape/hctsa-py/blob/master/PeripheryFunctions/wentropy.py
    if entType == 'shannon':
        x = np.power(x[ x != 0 ],2)
        return - np.sum(np.multiply(x,np.log(x)))

    elif entType == 'threshold':
        if additionalParameter is None or isinstance(additionalParameter, str):
            return None
        x = np.absolute(x)
        return np.sum((x > additionalParameter))

    elif entType == 'norm':
        if additionalParameter is None or isinstance(additionalParameter,str) or additionalParameter < 1:
            return None
        x = np.absolute(x)
        return np.sum(np.power(x, additionalParameter))

    elif entType == 'sure':
        if additionalParameter is None or isinstance(additionalParameter,str):
            return None

        N = len(x)
        x2 = np.square(x)
        t2 = additionalParameter**2
        xgt = np.sum((x2 > t2))
        xlt = N - xgt

        return N - (2*xlt) + (t2 *xgt) + np.sum(np.multiply(x2,(x2 <= t2)))

    elif entType == 'logenergy':
        x = np.square(x[x != 0])
        return np.sum(np.log(x))

    else:
        print("invalid entropy type")
        return None

def MSEnt(y : list, scaleRange = None, m = 2, r = 0.15, preProcessHow = None):
    """
    Multiscale entropy of a time series.
    """
    if scaleRange is None:
        scaleRange = range(1, 11)
    minTsLength = 20
    numScales = len(scaleRange)

    if preProcessHow is not None:
        y = BF_zscore(BF_PreProcess(y, preProcessHow))
    
    # Coarse-graining across scales
    y_cg = []
    for i in range(numScales):
        buffer_size = scaleRange[i]
        y_buffer = BF_MakeBuffer(y, buffer_size)
        y_cg.append(np.mean(y_buffer, axis=1))
    
    # Run sample entropy for each m and r value at each scale
    samp_ens = np.zeros(numScales)
    for si in range(numScales):
        if len(y_cg[si]) >= minTsLength:
            samp_en_struct = SampEn(y_cg[si], m, r)
            samp_ens[si] = samp_en_struct[f'sampen{m}']
        else:
            samp_ens[si] = np.nan

    # Outputs: multiscale entropy
    if np.all(np.isnan(samp_ens)):
        if preProcessHow:
            pp_text = f"after {preProcessHow} pre-processing"
        else:
            pp_text = ""
        print(f"Warning: Not enough samples ({len(y)} {pp_text}) to compute SampEn at multiple scales")
        return {'out': np.nan}

    # Output raw values
    out = {f'sampen_s{scaleRange[i]}': samp_ens[i] for i in range(numScales)}

     # Summary statistics of the variation
    max_samp_en = np.nanmax(samp_ens)
    max_ind = np.nanargmax(samp_ens)
    min_samp_en = np.nanmin(samp_ens)
    min_ind = np.nanargmin(samp_ens)

    out.update({
        'maxSampEn': max_samp_en,
        'maxScale': scaleRange[max_ind],
        'minSampEn': min_samp_en,
        'minScale': scaleRange[min_ind],
        'meanSampEn': np.nanmean(samp_ens),
        'stdSampEn': np.nanstd(samp_ens, ddof=1),
        'cvSampEn': np.nanstd(samp_ens, ddof=1) / np.nanmean(samp_ens),
        'meanch': np.nanmean(np.diff(samp_ens))
    })

    return out

def SampEn(y : list, M : int = 2, r : Union[None, float] = None, preProcessHow : Union[None, 'str'] = None):
    """
    Sample Entropy of a time series

    Uses an adaptation of SampEn(m, r) from PhysioNet.

    The publicly-available PhysioNet Matlab code, sampenc (renamed here to
    PN_sampenc) is available from:
    http://www.physionet.org/physiotools/sampen/matlab/1.1/sampenc.m

    cf. "Physiological time-series analysis using approximate entropy and sample
    entropy", J. S. Richman and J. R. Moorman, Am. J. Physiol. Heart Circ.
    Physiol., 278(6) H2039 (2000).

    This function can also calculate the SampEn of successive increments of time
    series, i.e., using an incremental differencing pre-processing, as
    used in the so-called Control Entropy quantity:

    "Control Entropy: A complexity measure for nonstationary signals"
    E. M. Bollt and J. Skufca, Math. Biosci. Eng., 6(1) 1 (2009).

    Parameters:
    -----------
    y (array-like):
        the input time series
    M (int, optional): 
        the embedding dimension
    r (float, optional): 
        the threshold
    preProcessHow (str, optional):
    (i) 'diff1', incremental differencing (as per 'Control Entropy').
    
    Returns:
    --------
    dict :
        A dictionary of sample entropy and quadratic sample entropy
    """
    if r is None:
        r = 0.1 * np.std(y, ddof=1)
    if preProcessHow is not None:
        y = BF_PreProcess(y, preProcessHow)
    
    out = {}
    sampEn, _, _, _ = PN_sampenc(y, M+1, r=r)
    # compute outputs 
    for i in range(len(sampEn)):
        out[f"sampen{i}"] = sampEn[i]
        # Quadratic sample entropy (QSE), Lake (2006):
        # (allows better comparison across r values)
        out[f"quadSampEn{i}"] = sampEn[i] + np.log(2*r)
    
    if M > 1:
        out['meanchsampen'] = np.mean(np.diff(sampEn))

    return out

def PermEn(y : list, m : int = 2, tau : Union[int, str] = 1):
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
        tau = FirstCrossing(y, 'ac', 0, 'discrete')
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

def _perm_entropy_all(x : list, order : int = 3, delay : int =1, normalize : bool = False, return_normedCounts : bool = False):
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
    
# def MEnt(y : list, scaleRange = None, m = 2, r = 0.15, preProcessHow = None):
#     """
#     Multiscale entropy of a time series.
#     """
#     if scaleRange is None:
#         scaleRange = range(1, 11)
#     minTsLength = 20
#     numScales = len(scaleRange)

#     if preProcessHow is not None:
#         y = BF_zscore(BF_PreProcess(y, preProcessHow))
    
#     # Coarse-graining across scales
#     y_cg = []
#     for i in range(numScales):
#         buffer_size = scaleRange[i]
#         y_buffer = BF_MakeBuffer(y, buffer_size)
#         y_cg.append(np.mean(y_buffer, axis=1))
    
#     # Run sample entropy for each m and r value at each scale
#     samp_ens = np.zeros(numScales)
#     for si in range(numScales):
#         if len(y_cg[si]) >= minTsLength:
#             samp_en_struct = EN_SampEn(y_cg[si], m, r)
#             samp_ens[si] = samp_en_struct[f'sampen{m}']
#         else:
#             samp_ens[si] = np.nan

#     # Outputs: multiscale entropy
#     if np.all(np.isnan(samp_ens)):
#         if preProcessHow:
#             pp_text = f"after {preProcessHow} pre-processing"
#         else:
#             pp_text = ""
#         print(f"Warning: Not enough samples ({len(y)} {pp_text}) to compute SampEn at multiple scales")
#         return {'out': np.nan}

#     # Output raw values
#     out = {f'sampen_s{scaleRange[i]}': samp_ens[i] for i in range(numScales)}

#      # Summary statistics of the variation
#     max_samp_en = np.nanmax(samp_ens)
#     max_ind = np.nanargmax(samp_ens)
#     min_samp_en = np.nanmin(samp_ens)
#     min_ind = np.nanargmin(samp_ens)

#     out.update({
#         'maxSampEn': max_samp_en,
#         'maxScale': scaleRange[max_ind],
#         'minSampEn': min_samp_en,
#         'minScale': scaleRange[min_ind],
#         'meanSampEn': np.nanmean(samp_ens),
#         'stdSampEn': np.nanstd(samp_ens, ddof=1),
#         'cvSampEn': np.nanstd(samp_ens, ddof=1) / np.nanmean(samp_ens),
#         'meanch': np.nanmean(np.diff(samp_ens))
#     })

#     return out

def LZcomplexity(x : list, nbits : int = 2, preProc : str = 'none') -> float:
    # n is number of bits to encode into
    if preProc == 'diff':
        x = BF_zscore(np.diff(x))
    x = np.array(x).flatten()
    nx = x.size
    # Add small noise to handle ties
    noise = np.finfo(float).eps * np.random.randn(nx)
    x_noisy = x + noise
    xi = np.argsort(x_noisy)
    y = np.arange(1, nx + 1)
    y = np.floor(y * (nbits / (nx + 1))).astype(int)
    x_sorted_indices = xi
    x[x_sorted_indices] = y

    return _lempel_ziv_complexity(x)

def _lempel_ziv_complexity(data):
    # adapted from Michael Small's code (add reference)
    # Convert data to symbols
    s = [int(np.floor(x)) + 1 for x in data]
    bins = max(s) if s else 0  # bins is the maximum symbol value
    n = len(s)
    
    if n == 0:
        return 0.0
    
    c = 1  
    ns = 1  
    nq = 1  
    k = 2  
    
    while k < n:
        is_substring = False
        current_sub = s[ns:ns + nq]
        max_i = ns - nq
        for i in range(0, max_i + 1):
            # Check if the substring starting at i matches current_sub
            if s[i:i + nq] == current_sub:
                is_substring = True
                break
        
        if is_substring:
            nq += 1
        else:
            c += 1
            ns += nq
            nq = 1
        
        k += 1
    
    # Handle potential division by zero if bins is 1
    if bins == 0:
        return 0.0
    try:
        normalized = (c * np.log(n)) / (n * np.log(bins))
    except ZeroDivisionError:
        # If bins is 1, log(1) is 0, leading to division by zero
        normalized = float('inf')
    
    return normalized

def CID(y : list):
    """
    Simple complexity measure of a time series.

    Estimates of 'complexity' of a time series as the stretched-out length of the
    lines resulting from a line-graph of the time series.

    Parameters:
    y (array-like): the input time series

    Returns:
    out (dict): dictionary of estimates.
    """
    CE1 = np.sqrt(np.mean(np.power(np.diff(y),2)))
    CE2 = np.mean(np.sqrt(1 + np.power(np.diff(y),2)))

    minCE1 = f_CE1(np.sort(y))
    minCE2 = f_CE2(np.sort(y))

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':CE1,'CE2':CE2,'minCE1':minCE1,'minCE2':minCE2,
            'CE1_norm':CE1_norm,'CE2_norm':CE2_norm}

    return out

def f_CE1(y):
    # Original definition (in Table 2 of paper cited above)
    # sum -> mean to deal with non-equal time-series lengths
    # (now scales properly with length)
    return np.sqrt(np.mean(np.power(np.diff(y),2)))

def f_CE2(y):
    # Definition corresponding to the line segment example in Fig. 9 of the paper
    # cited above (using Pythagoras's theorum):
    return np.mean(np.sqrt(1 + np.power(np.diff(y),2)))

def ApEN(y : list, mnom : int = 1, rth : float = 0.2):
    """
    Approximate Entropy of a time series

    ApEn(m,r).

    Parameters:
    -----------
    y : array-like
        The input time series
    mnom : int, optional
        The embedding dimension (default is 1)
    rth : float, optional
        The threshold for judging closeness/similarity (default is 0.2)

    Returns:
    --------
    float
        The Approximate Entropy value

    References:
    -----------
    S. M. Pincus, "Approximate entropy as a measure of system complexity",
    P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)

    For more information, cf. http://physionet.org/physiotools/ApEn/
    """
    r = rth * np.std(y, ddof=1) # threshold of similarity
    N = len(y) # time series length
    phi = np.zeros(2) # phi[0] = phi_m, phi[1] = phi_{m+1}

    for k in range(2):
        m = mnom+k # pattern length
        C = np.zeros(N - m + 1)
        # define the matrix x, containing subsequences of u
        x = np.zeros((N-m+1, m))

        # Form vector sequences x from the time series y
        x = np.array([y[i:i+m] for i in range(N - m + 1)])
        
        for i in range(N - m + 1):
            # Calculate the number of x[j] within r of x[i]
            d = np.abs(x - x[i])
            if m > 1:
                d = np.max(d, axis=1)
            C[i] = np.sum(d <= r) / (N - m + 1)

        phi[k] = np.mean(np.log(C))

    return phi[0] - phi[1]
