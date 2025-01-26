import numpy as np

def EN_ApEN(y, mnom = 1, rth = 0.2):
    """
    Approximate Entropy of a time series

    ApEn(m,r).

    Parameters:
    -----------
    y : array-like
        The input time series
    mnom : int, optional
        The embedding dimension (default is 1)
    rth : float, optional
        The threshold for judging closeness/similarity (default is 0.2)

    Returns:
    --------
    float
        The Approximate Entropy value

    References:
    -----------
    S. M. Pincus, "Approximate entropy as a measure of system complexity",
    P. Natl. Acad. Sci. USA, 88(6) 2297 (1991)

    For more information, cf. http://physionet.org/physiotools/ApEn/
    """
    r = rth * np.std(y, ddof=1) # threshold of similarity
    N = len(y) # time series length
    phi = np.zeros(2) # phi[0] = phi_m, phi[1] = phi_{m+1}

    for k in range(2):
        m = mnom+k # pattern length
        C = np.zeros(N - m + 1)
        # define the matrix x, containing subsequences of u
        x = np.zeros((N-m+1, m))

        # Form vector sequences x from the time series y
        x = np.array([y[i:i+m] for i in range(N - m + 1)])
        
        for i in range(N - m + 1):
            # Calculate the number of x[j] within r of x[i]
            d = np.abs(x - x[i])
            if m > 1:
                d = np.max(d, axis=1)
            C[i] = np.sum(d <= r) / (N - m + 1)

        phi[k] = np.mean(np.log(C))

    return phi[0] - phi[1]
