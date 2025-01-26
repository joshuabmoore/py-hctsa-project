import numpy as np

def DN_Withinp(x, p = 1, meanOrMedian = 'mean'):
    """
    Proportion of data points within p standard deviations of the mean or median.

    Parameters:
    -----------
    x (array-like): The input data vector
    p (float): The number (proportion) of standard deviations
    meanOrMedian (str): Whether to use units of 'mean' and standard deviation,
                          or 'median' and rescaled interquartile range

    Returns:
    --------
    float: The proportion of data points within p standard deviations

    Raises:
    ValueError: If mean_or_median is not 'mean' or 'median'
    """
    x = np.asarray(x)
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        iqr_val = np.percentile(x, 75, method='hazen') - np.percentile(x, 25, method='hazen')
        sig = 1.35 * iqr_val
    else:
        raise ValueError(f"Unknown setting: '{meanOrMedian}'")

    # The withinp statistic:
    return np.sum((x >= mu - p * sig) & (x <= mu + p * sig)) / N
