import numpy as np
import math
from scipy.fft import fft
from scipy import signal

def MD_hrv_classic(y):
    """
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

    out["SD1"] = 1/math.sqrt(2) * rmssd * 1000
    out["SD2"] = math.sqrt(2 * sigma**2 - (1/2) * rmssd**2) * 1000

    return out
