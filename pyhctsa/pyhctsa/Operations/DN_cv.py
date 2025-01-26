import warnings
import numpy as np

def DN_cv(x, k = 1):
    """
    Coefficient of variation

    Coefficient of variation of order k is sigma^k / mu^k (for sigma, standard
    deviation and mu, mean) of a data vector, x

    Parameters:
    ----------
    x (array-like): The input data vector
    k (int, optional): The order of coefficient of variation (k = 1 is default)

    Returns:
    -------
    float: The coefficient of variation of order k
    """
    if not isinstance(k, int) or k < 0:
        warnings.warn('k should probably be a positive integer')
        # Carry on with just this warning, though
    
    # Compute the coefficient of variation (of order k) of the data
    return (np.std(x, ddof=1) ** k) / (np.mean(x) ** k)
