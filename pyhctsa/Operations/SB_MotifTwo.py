import numpy as np
from PeripheryFunctions.BF_Binarize import BF_Binarize
import warnings

def SB_MotifTwo(y, binarizeHow = 'diff'):
    """
    SB_MotifTwo  Local motifs in a binary symbolization of the time series.

    Coarse-graining is performed by a given binarization method.

    Parameters
    ----------
    y : array_like
        The input time series.
    binarize_how : str, optional
        The binary transformation method (default is 'diff'):
        - 'diff': Incremental time-series increases are encoded as 1, and decreases as 0.
        - 'mean': Time-series values above the mean are given 1, and those below the mean are 0.
        - 'median': Time-series values above the median are given 1, and those below the median are 0.

    Returns
    -------
    dict
        A dictionary containing the probabilities of words in the binary alphabet of lengths 1, 2, 3, and 4, 
        and their entropies.
    """
    # Generate a binarized version of the input time series
    yBin = BF_Binarize(y, binarizeHow)

    # Define the length of the new, symbolized sequence, N
    N = len(yBin)

    if N < 5:
        warnings.warn("Time series too short!")
        return np.Nan
    
    # Binary sequences of length 1
    r1 = (yBin == 1) # 1
    r0 = (yBin == 0) # 0

    # ------ Record these -------
    # (Will be dependent outputs since signal is binary, sum to 1)
    # (Default hctsa library measures just the u output: up)
    out = {}
    out['u'] = np.mean(r1) # proportion 1 (corresponds to a movement up for 'diff')
    out['d'] = np.mean(r0) # proportion 0 (corresponds to a movement down for 'diff')
    pp = [out['d'], out['u']]
    out['h'] = f_entropy(pp)

    # Binary sequences of length 2:
    r1 = r1[:-1]
    r0 = r0[:-1]

    r00 = np.logical_and(r0, yBin[1:] == 0)
    r01 = np.logical_and(r0, yBin[1:] == 1)
    r10 = np.logical_and(r1, yBin[1:] == 0)
    r11 = np.logical_and(r1, yBin[1:] == 1)

    out['dd'] = np.mean(r00)  # down, down
    out['du'] = np.mean(r01)  # down, up
    out['ud'] = np.mean(r10)  # up, down
    out['uu'] = np.mean(r11)  # up, up

    pp = [out['dd'], out['du'], out['ud'], out['uu']]
    out['hh'] = f_entropy(pp)

    # -----------------------------
    # Binary sequences of length 3:
    # -----------------------------
    # Make sure ranges are valid for looking at the next one
    r00 = r00[:-1]
    r01 = r01[:-1]
    r10 = r10[:-1]
    r11 = r11[:-1]

    # 000
    r000 = np.logical_and(r00, yBin[2:] == 0)
    # 001 
    r001 = np.logical_and(r00, yBin[2:] == 1)
    r010 = np.logical_and(r01, yBin[2:] == 0)
    r011 = np.logical_and(r01, yBin[2:] == 1)
    r100 = np.logical_and(r10, yBin[2:] == 0)
    r101 = np.logical_and(r10, yBin[2:] == 1)
    r110 = np.logical_and(r11, yBin[2:] == 0)
    r111 = np.logical_and(r11, yBin[2:] == 1)

    # ----- Record these -----
    out['ddd'] = np.mean(r000)
    out['ddu'] = np.mean(r001)
    out['dud'] = np.mean(r010)
    out['duu'] = np.mean(r011)
    out['udd'] = np.mean(r100)
    out['udu'] = np.mean(r101)
    out['uud'] = np.mean(r110)
    out['uuu'] = np.mean(r111)

    ppp = [out['ddd'], out['ddu'], out['dud'], out['duu'], out['udd'], out['udu'], out['uud'], out['uuu']]
    out['hhh'] = f_entropy(ppp)

    # -------------------
    # 4
    # -------------------
    # Make sure ranges are valid for looking at the next one

    r000 = r000[:-1]
    r001 = r001[:-1]
    r010 = r010[:-1]
    r011 = r011[:-1]
    r100 = r100[:-1]
    r101 = r101[:-1]
    r110 = r110[:-1]
    r111 = r111[:-1]

    r0000 = np.logical_and(r000, yBin[3:] == 0)
    r0001 = np.logical_and(r000, yBin[3:] == 1)
    r0010 = np.logical_and(r001, yBin[3:] == 0)
    r0011 = np.logical_and(r001, yBin[3:] == 1)
    r0100 = np.logical_and(r010, yBin[3:] == 0)
    r0101 = np.logical_and(r010, yBin[3:] == 1)
    r0110 = np.logical_and(r011, yBin[3:] == 0)
    r0111 = np.logical_and(r011, yBin[3:] == 1)
    r1000 = np.logical_and(r100, yBin[3:] == 0)
    r1001 = np.logical_and(r100, yBin[3:] == 1)
    r1010 = np.logical_and(r101, yBin[3:] == 0)
    r1011 = np.logical_and(r101, yBin[3:] == 1)
    r1100 = np.logical_and(r110, yBin[3:] == 0)
    r1101 = np.logical_and(r110, yBin[3:] == 1)
    r1110 = np.logical_and(r111, yBin[3:] == 0)
    r1111 = np.logical_and(r111, yBin[3:] == 1)

    # ----- Record these -----
    out['dddd'] = np.mean(r0000)
    out['dddu'] = np.mean(r0001)
    out['ddud'] = np.mean(r0010)
    out['dduu'] = np.mean(r0011)
    out['dudd'] = np.mean(r0100)
    out['dudu'] = np.mean(r0101)
    out['duud'] = np.mean(r0110)
    out['duuu'] = np.mean(r0111)
    out['uddd'] = np.mean(r1000)
    out['uddu'] = np.mean(r1001)
    out['udud'] = np.mean(r1010)
    out['uduu'] = np.mean(r1011)
    out['uudd'] = np.mean(r1100)
    out['uudu'] = np.mean(r1101)
    out['uuud'] = np.mean(r1110)
    out['uuuu'] = np.mean(r1111)

    pppp = [out['dddd'], out['dddu'], out['ddud'], out['dduu'], out['dudd'], out['dudu'], out['duud'], out['duuu'],
            out['uddd'], out['uddu'], out['udud'], out['uduu'], out['uudd'], out['uudu'], out['uuud'], out['uuuu']]
    out['hhhh'] = f_entropy(pppp)

    return out

# helper function 
def f_entropy(x):
    """Entropy of a set of counts, log(0) = 0"""
    return -np.sum(x[x > 0] * np.log(x[x > 0]))
