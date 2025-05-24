# medical 
import numpy as np
from scipy import signal

def RawHRVmeas(x : list) -> dict:
    """
    MD_rawHRVmeas computes Poincare plot measures used in HRV (Heart Rate Variability) analysis.
    
    This function computes the triangular histogram index and Poincare plot measures for a time
    series assumed to measure sequences of consecutive RR intervals in milliseconds. It is not 
    suitable for other types of time series.

    Due to differences in how MATLAB implements its histogram function (histcounts) and how
    numpy implements its histogram function (histogram), expect a discrepancy for the feature:
    `tri`.

    Parameters:
    ----------
    x : array_like
        A time series assumed to measure sequences of consecutive RR intervals in milliseconds.

    Returns:
    -------
    out : dict
        A dictionary containing the following keys:
        - 'tri10': Triangular histogram index with 10 bins
        - 'tri20': Triangular histogram index with 20 bins
        - 'trisqrt': Triangular histogram index with bins calculated using the square root method
        - 'SD1': Standard deviation of the Poincare plot's minor axis
        - 'SD2': Standard deviation of the Poincare plot's major axis
    """

    N = len(x)
    out = {}

    # triangular histogram index
    hist_counts10, _ = np.histogram(x, 10)
    out['tri10'] = N/np.max(hist_counts10)
    hist_counts20, _ = np.histogram(x, 20)
    out['tri20'] = N/np.max(hist_counts20)
    # MATLAB histcounts returns wrong number of bins for sqrt rule. This should be the correct num of bins...
    hist_counts_sqrt, _ = np.histogram(x, bins=int(np.ceil(np.sqrt(N))))
    out['trisqrt'] = N/np.max(hist_counts_sqrt)

    # Poincare plot measures
    diffx = np.diff(x)
    out['SD1'] = 1/np.sqrt(2) * np.std(diffx, ddof=1) * 1000
    out['SD2'] = np.sqrt(2 * np.var(x, ddof=1) - (1/2) * np.std(diffx, ddof=1)**2) * 1000

    return out

def Polvar(x : list, d : float = 1, D : int = 6):
    """
    The POLVARd measure of a time series.
    Measures the probability of obtaining a sequence of consecutive ones or zeros.

    The first mention may be in Wessel et al., PRE (2000), called Plvar
    cf. "Short-term forecasting of life-threatening cardiac arrhythmias based on
    symbolic dynamics and finite-time growth rates",
        N. Wessel et al., Phys. Rev. E 61(1) 733 (2000)
    
    Although the original measure used raw thresholds, d, on RR interval sequences
    (measured in milliseconds), this code can be applied to general z-scored time
    series. So now d is not the time difference in milliseconds, but in units of
    std.
    
    The measure was originally applied to sequences of RR intervals and this code
    is heavily derived from that provided by Max A. Little, January 2009.
    cf. http://www.maxlittle.net/

    Parameters:
    -----------
    x : array_like
        Input time series
    d : float
        The symbolic coding (amplitude) difference
    D : int
        The word length.

    Returns:
    --------
    p : float
        Probability of obtaining a sequence of consecutive ones/zeros.
    """

    dx = np.abs(np.diff(x)) # abs diff in consecutive values of the time series
    N = len(dx) # number of diffs in the input time series

    # binary representation of time series based on consecutive changes being greater than d/1000...
    xsym = dx >= d # consec. diffs exceed some threshold, d
    zseq = np.zeros(D)
    oseq = np.ones(D)

    # search for D consecutive zeros/ones
    i = 1
    pc = 0

    # seqcnt = 0
    while i <= (N-D):
        xseq = xsym[i:(i+D)]
        if (np.sum(xseq == zseq) == D) or (np.sum(xseq == oseq) == D):
            pc += 1
            i += D
        else:
            i += 1
    
    p = pc / N

    return p 

def PNN(x : list) -> dict:
    """
    MD_pNN: pNNx measures of heart rate variability.

    Applies pNNx measures to time series assumed to represent sequences of
    consecutive RR intervals measured in milliseconds.

    This code is derived from MD_hrv_classic.m because it doesn't make medical
    sense to do PNN on a z-scored time series. But now PSD doesn't make too much sense, 
    so we just evaluate the pNN measures.

    Parameters:
    -----------
    x (array-like): Input time series.

    Returns:
    --------
    dict: A dictionary containing the pNNx measures.
    """

    diffx = np.diff(x)
    N = len(x)

    # Calculate pNNx percentage

    Dx = np.abs(diffx) * 1000 # assume milliseconds as for RR intervals
    pnns = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    out = {} # dict used for output in place of MATLAB struct

    for x in pnns:
        out["pnn" + str(x) ] = sum(Dx > x) / (N-1)

    return out

def HRV_classic(y : list) -> dict:
    """
    Classic heart rate variability (HRV) statistics.

    Due to differences in how MATLAB implements its histogram function (histcounts) and how
    numpy implements its histogram function (histogram), expect a discrepancy for the feature:
    `tri`.
    """

    # Standard defaults
    diffy = np.diff(y)
    N = len(y)

    # ------------------------------------------------------------------------------
    # Calculate pNNx percentage
    # ------------------------------------------------------------------------------
    # pNNx: recommendation as per Mietus et. al. 2002, "The pNNx files: ...", Heart
    # strange to do this for a z-scored time series...
    Dy = np.abs(diffy)
    PNNxfn = lambda x : np.mean(Dy > x/1000)

    out = {}

    out['pnn5'] = PNNxfn(5) # 0.0055*sigma
    out['pnn10'] = PNNxfn(10) # 0.01*sigma
    out['pnn20'] = PNNxfn(20) # 0.02*sigma
    out['pnn30'] = PNNxfn(30) # 0.03*sigma
    out['pnn40'] = PNNxfn(40) # 0.04*sigma

    # ------------------------------------------------------------------------------
    # Calculate PSD
    # ------------------------------------------------------------------------------

    nfft = max(256, 2**int(np.ceil(np.log2((N)))))
    F, Pxx = signal.periodogram(y, window = np.hanning(len(y)), detrend=False, scaling='density', fs=2*np.pi, nfft=nfft)

    # Calculate spectral measures such as subband spectral power percentage, LF/HF ratio etc.
    LF_lo = 0.04 # /pi -- fraction of total power (max F is pi)
    LF_hi = 0.15
    HF_lo = 0.15
    HF_hi = 0.4

    fbinsize = F[1] - F[0]

    # calculating indl, indh, indv; needed for loop for python implementation
    indl = []
    for x in F:
        if x >= LF_lo and x <= LF_hi:
            indl.append(1)
        else :
            indl.append(0)
    indh = []
    for x in F:
        if x >= HF_lo and x <= HF_hi:
            indh.append(1)
        else:
            indh.append(0)

    indv = []
    for x in F:
        if x <= LF_lo:
            indv.append(1)
        else :
            indv.append(0)

    indlPxx = []
    for i in range(0, len(Pxx)):
        if indl[i] == 1:
            indlPxx.append(Pxx[i])
    lfp = fbinsize * np.sum(indlPxx)

    indhPxx = []
    for i in range(0, len(Pxx)):
        if indh[i] == 1:
            indhPxx.append(Pxx[i])
    hfp = fbinsize * np.sum(indhPxx)

    indvPxx = []
    for i in range(0, len(Pxx)):
        if indv[i] == 1:
            indvPxx.append(Pxx[i])
    vlfp = fbinsize * np.sum(indvPxx)

    out['lfhf'] = lfp / hfp
    total = fbinsize * np.sum(Pxx)
    out['vlf'] = vlfp/total * 100
    out['lf'] = lfp/total * 100
    out['hf'] = hfp/total * 100

    # Triangular histogram index
    numBins = 10
    hist = np.histogram(y, bins=numBins)
    out['tri'] = len(y)/np.max(hist[0])

    # Poincare plot measures:
    # cf. "Do Existing Measures ... ", Brennan et. al. (2001), IEEE Trans Biomed Eng 48(11)
    rmssd = np.std(diffy, ddof=1)
    sigma = np.std(y, ddof=1)

    out["SD1"] = 1/np.sqrt(2) * rmssd * 1000
    out["SD2"] = np.sqrt(2 * sigma**2 - (1/2) * rmssd**2) * 1000

    return out
