# nonlinear
import numpy as np
from nolitsa import dimension
from typing import Union
from IN import FirstCrossing, FirstMin


def FNN(y : list, maxdim : int = 10, tau : Union[str, int] = 1, th : Union[int, float] = 5, kth : Union[int, float] = 1, justBest : Union[bool, int] = 0, bestp : float = 0.1) -> dict:
    """
    False nearest neighbors of a time series.
    Currently a stand-in for NL_TISEAN_fnn, NL_MS_fnn and NL_TISEAN_fnn
    """
    if isinstance(tau, str):
        if tau == 'mi': # use first minimum of AMI
            tau = FirstMin(y, 'mi')
        elif tau == 'ac': # use first zero-crossing of autocorrelation function
            tau = FirstCrossing(y, 'ac', 0, 'discrete')
        else:
            raise ValueError(f"Invalid method for tau: {tau}. Choose either 'mi' or 'ac'.")
    
    if np.isnan(tau):
        print("Time series too short for fnn. Returning NaN value.")
        return np.nan

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
