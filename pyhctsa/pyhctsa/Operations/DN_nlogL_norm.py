import numpy as np

def DN_nlogL_norm(y):
    """
    Negative log likelihood of data coming from a Gaussian distribution.

    This function fits a Gaussian distribution to the data and returns the negative
    log likelihood of the data coming from that Gaussian distribution.

    Parameters:
    y (array-like): A vector of data

    Returns:
    float: The negative log likelihood per data point
    """
    # Convert input to numpy array
    y = np.asarray(y)

    # Fit a Gaussian distribution to the data (mimicking MATLAB's normfit)
    mu = np.mean(y)
    sigma = np.std(y, ddof=1)  # ddof=1 for sample standard deviation

    # Compute the negative log-likelihood (mimicking MATLAB's normlike)
    n = len(y)
    nlogL = (n/2) * np.log(2*np.pi) + n*np.log(sigma) + np.sum((y - mu)**2) / (2*sigma**2)

    # Return the average negative log-likelihood
    return nlogL / n
