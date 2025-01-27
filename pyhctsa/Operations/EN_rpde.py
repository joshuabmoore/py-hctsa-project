import numpy as np
from PeripheryFunctions.BF_Embed import BF_Embed
from pyrpde import rpde

def EN_rpde(x, m = 2, tau = 1, epsilon = 0.12, T_max = -1):

    _, rpd = rpde(np.array(x, dtype=np.float32), tau=tau, dim=m, epsilon=epsilon, tmax=len(x)-1)

    if T_max > - 1:
        rpd = rpd[:T_max]
    rpd = rpd/np.sum(rpd)
    N = len(rpd)
    ip = rpd > 0
    H = -np.sum(rpd[ip]*np.log(rpd[ip]))
    H_norm = H/np.log(N) # % log(N) is the H for an i.i.d. process
    out = {}
    out["H"] = H
    out["H_norm"] = H_norm

    out["propNonZero"] = np.mean(rpd > 0)
    out["meanNonZero"] = np.mean(rpd[rpd>0])*N 
    out["maxRPD"] = np.max(rpd)*N

    return out
