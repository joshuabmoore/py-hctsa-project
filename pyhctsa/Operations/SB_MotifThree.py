import numpy as np
from Operations.SB_CoarseGrain import SB_CoarseGrain

def SB_MotifThree(y, cgHow = 'quantile'):
    """
    Motifs in a coarse-graining of a time series to a 3-letter alphabet.

    Parameters:
    -----------
    y : np.ndarray
        Time series to analyze.
    cg_how : {'quantile', 'diffquant'}, optional
        The coarse-graining method to use:
        - 'quantile': equiprobable alphabet by time-series value
        - 'diffquant': equiprobably alphabet by time-series increments
        Default is 'quantile'.

    Returns:
    --------
    Dict[str, float]
        Statistics on words of length 1, 2, 3, and 4.
    """

    # Coarse-grain the data y -> yt
    numLetters = 3
    if cgHow == 'quantile':
        yt = SB_CoarseGrain(y, 'quantile', numLetters)
    elif cgHow == 'diffquant':
        yt = SB_CoarseGrain(np.diff(y), 'quantile', numLetters)
    else:
        raise ValueError(f"Unknown coarse-graining method {cgHow}")

    # So we have a vectory yt with entries in {1, 2, 3}
    N = len(yt) # length of the symbolized sequence derived from the time series

    # ------------------------------------------------------------------------------
    # Words of length 1
    # ------------------------------------------------------------------------------
    out1 = np.zeros(3)
    r1 = [np.where(yt == i + 1)[0] for i in range(3)]
    for i in range(3):
        out1[i] = len(r1[i]) / N

    out = {
        'a': out1[0], 'b': out1[1], 'c': out1[2],
        'h': f_entropy(out1)
    }

    # ------------------------------------------------------------------------------
    # Words of length 2
    # ------------------------------------------------------------------------------

    r1 = [r[:-1] if len(r) > 0 and r[-1] == N - 1 else r for r in r1]
    out2 = np.zeros((3, 3))
    r2 = [[r1[i][yt[r1[i] + 1] == j + 1] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            out2[i, j] = len(r2[i][j]) / (N - 1)

    out.update({
        'aa': out2[0, 0], 'ab': out2[0, 1], 'ac': out2[0, 2],
        'ba': out2[1, 0], 'bb': out2[1, 1], 'bc': out2[1, 2],
        'ca': out2[2, 0], 'cb': out2[2, 1], 'cc': out2[2, 2],
        'hh': f_entropy(out2)
    })

    # ------------------------------------------------------------------------------
    # Words of length 3
    # ------------------------------------------------------------------------------

    r2 = [[r[:-1] if len(r) > 0 and r[-1] == N - 2 else r for r in row] for row in r2]
    out3 = np.zeros((3, 3, 3))
    r3 = [[[r2[i][j][yt[r2[i][j] + 2] == k + 1] for k in range(3)] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out3[i, j, k] = len(r3[i][j][k]) / (N - 2)

    out.update({f'{chr(97+i)}{chr(97+j)}{chr(97+k)}': out3[i, j, k] 
                for i in range(3) for j in range(3) for k in range(3)})
    out['hhh'] = f_entropy(out3)

    # ------------------------------------------------------------------------------
    # Words of length 4
    # ------------------------------------------------------------------------------

    r3 = [[[r[:-1] if len(r) > 0 and r[-1] == N - 3 else r for r in plane] for plane in cube] for cube in r3]
    out4 = np.zeros((3, 3, 3, 3))
    r4 = [[[[r3[i][j][k][yt[r3[i][j][k] + 3] == l + 1] for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    out4[i, j, k, l] = len(r4[i][j][k][l]) / (N - 3)

    out.update({f'{chr(97+i)}{chr(97+j)}{chr(97+k)}{chr(97+l)}': out4[i, j, k, l] 
                for i in range(3) for j in range(3) for k in range(3) for l in range(3)})
    out['hhhh'] = f_entropy(out4)

    return out

# helper function 
def f_entropy(x):
    """Entropy of a set of counts, log(0) = 0"""
    return -np.sum(x[x > 0] * np.log(x[x > 0]))
