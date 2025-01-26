import numpy as np
from PeripheryFunctions.BF_PreProcess import BF_PreProcess
from PeripheryFunctions.PN_sampenc import PN_sampenc

def EN_SampEn(y, M = 2, r = None, preProcessHow = None):
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
