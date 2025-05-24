import numpy as np
import warnings
from pyhctsa.Operations_OLD.BF_zscore import BF_zscore

def lempel_ziv_complexity(data):
    # adapted from Michael Small's code (add reference)
    # Convert data to symbols
    s = [int(np.floor(x)) + 1 for x in data]
    bins = max(s) if s else 0  # bins is the maximum symbol value
    n = len(s)
    
    if n == 0:
        return 0.0
    
    c = 1  
    ns = 1  
    nq = 1  
    k = 2  
    
    while k < n:
        is_substring = False
        current_sub = s[ns:ns + nq]
        max_i = ns - nq
        for i in range(0, max_i + 1):
            # Check if the substring starting at i matches current_sub
            if s[i:i + nq] == current_sub:
                is_substring = True
                break
        
        if is_substring:
            nq += 1
        else:
            c += 1
            ns += nq
            nq = 1
        
        k += 1
    
    # Handle potential division by zero if bins is 1
    if bins == 0:
        return 0.0
    try:
        normalized = (c * np.log(n)) / (n * np.log(bins))
    except ZeroDivisionError:
        # If bins is 1, log(1) is 0, leading to division by zero
        normalized = float('inf')
    
    return normalized

def EN_LZcomplexity(x : list, n : int = 2, preProc : str = 'none'):
    if preProc == 'diff':
        x = BF_zscore(np.diff(x))
    x = np.array(x).flatten()
    nx = x.size
    # Add small noise to handle ties
    noise = np.finfo(float).eps * np.random.randn(nx)
    x_noisy = x + noise
    xi = np.argsort(x_noisy)
    y = np.arange(1, nx + 1)
    y = np.floor(y * (n / (nx + 1))).astype(int)
    x_sorted_indices = xi
    x[x_sorted_indices] = y

    return lempel_ziv_complexity(x)
