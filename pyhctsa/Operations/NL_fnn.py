import numpy as np
from nolitsa import dimension

def NL_fnn(y, maxdim = 10, tau = 1, th = 5, kth = 1, justBest = 0, bestp = 0.1):
    dim = np.arange(1, maxdim+1)
    p = dimension.fnn(y, dim=dim, tau=tau, R=th, window=kth)[0]
    if justBest:
        out = dim[np.argwhere(p < bestp)[0]]
    else:
        out = {}
        for i in range(maxdim):
            out[f"pfnn_{i+1}"] = p[i]

        out['meanpfnn'] = np.mean(p)
        out['stdpfnn'] = np.std(p, ddof=1)

        # find embedding dimension for the first time p goes under x%
        out['firstunder02'] = dim[np.argwhere(p < 0.2)[0]][0]
        out['firstunder01'] = dim[np.argwhere(p < 0.1)[0]][0]
        out['firstunder005'] = dim[np.argwhere(p < 0.05)[0]][0]
        out['firstunder002'] = dim[np.argwhere(p < 0.02)[0]][0]
        out['firstunder001'] = dim[np.argwhere(p < 0.01)[0]][0]

        out['max1stepchange'] = np.max(np.abs(np.diff(p)))

    return out
