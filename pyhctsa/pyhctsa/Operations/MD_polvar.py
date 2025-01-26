import numpy as np

def MD_polvar(x, d = 1, D = 6):
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
