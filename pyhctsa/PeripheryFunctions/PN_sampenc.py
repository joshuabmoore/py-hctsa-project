import numpy as np
from numba import jit

@jit(nopython=True)
def PN_sampenc(y, M = 1, r = None, justM = False):
    """
    Calculate Sample Entropy

    Parameters:
    y (array-like): Input time-series data
    M (int): Maximum template length (embedding dimension)
    r (float, optional): Matching tolerance level. If None, defaults to 0.1 * std(y)
    justM (bool, optional): If True, return e just for the given M, not for all m up to it

    Returns:
    tuple: (e, p, A, B)
        e: Sample entropy estimates for m=0,1,...,M-1
        p: Probabilities
        A: Number of matches for m=1,...,M
        B: Number of matches for m=1,...,M excluding last point
    """
    if r is None:
        # need to manually compute std so everything works with numba...
        ddof = 1
        mean_val = np.sum(y) / len(y)
        squared_diff_sum = np.sum((y - mean_val)**2)
        variance = squared_diff_sum / (len(y) - ddof)
        std_val = np.sqrt(variance)
        r = 0.1 * std_val
    
    N = len(y)
    lastrun = np.zeros(N)
    run = np.zeros(N)
    A = np.zeros(M)
    B = np.zeros(M)
    p = np.zeros(M)
    e = np.zeros(M)

    # get counting 
    for i in range(N-1): # go through each point in the time series, counting matches
        y1 = y[i]
        for jj in range(N-i-1):
            # compare to future index, j
            j = i + jj + 1
            # this future point, j, matches the time series value at i
            if np.abs(y[j]-y1) < r:
                run[jj] = lastrun[jj] + 1 # increase run count for this lag
                M1 = int(min(M, run[jj]))
                for m in range(M1):
                    A[m] += 1
                    if j < N - 1:
                        B[m] += 1
            else:
                run[jj] = 0
        for j in range(N-i-1):
            lastrun[j] = run[j]
        
    # Calculate for m = 1
    NN = N*(N-1)/2
    p[0] = A[0]/NN
    e[0] = -np.log(p[0])

    # calculate for m > 1, up to M
    for m in range(1, M):
        p[m] = A[m]/B[m-1]
        e[m] = -np.log(p[m])

    # Flag to output the entropy and probability just at the maximum requested m
    if justM == True:
        return np.array([e[-1]]), np.array([p[-1]]), A, B
    else:
        return e, p, A, B
