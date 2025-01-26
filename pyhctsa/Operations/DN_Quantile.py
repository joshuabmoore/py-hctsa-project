import numpy as np

def DN_Quantile(y, p=0.5):
    """
    Calculates the quantile value at a specified proportion, p.

    Parameters:
    y (array-like): The input data vector
    p (float): The quantile proportion (default is 0.5, which is the median)

    Returns:
    float: The calculated quantile value

    Raises:
    ValueError: If p is not a number between 0 and 1
    """
    if p == 0.5:
        print("Using quantile p = 0.5 (median) by default")
    
    if not isinstance(p, (int, float)) or p < 0 or p > 1:
        raise ValueError("p must specify a proportion, in (0,1)")
    
    return np.quantile(y, p, method='hazen')
