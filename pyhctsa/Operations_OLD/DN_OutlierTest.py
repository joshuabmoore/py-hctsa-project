import numpy as np 

def DN_OutlierTest(y, p = 2, justMe = None):
    """
    DN_OutlierTest    How distributional statistics depend on distributional outliers.

    Removes the p% of highest and lowest values in the time series (i.e., 2*p%
    removed from the time series in total) and returns the ratio of either the
    mean or the standard deviation of the time series, before and after this
    transformation.

    Parameters:
    -----------
    y (array-like): The input data vector
    p (float): The percentage of values to remove beyond upper and lower percentiles (default: 2)
    justMe (str, optional): Just returns a number:
                            'mean' -- returns the mean of the middle portion of the data
                            'std' -- returns the std of the middle portion of the data

    Returns:
    --------
    dict or float: A dictionary with 'mean' and 'std' keys, or a single value if justMe is specified
    """

    # mean of the middle (100-2*p)% of the data
    y = np.array(y)
    lower_bound = np.percentile(y, p, method='hazen')
    upper_bound = np.percentile(y, (100 - p), method='hazen')
    
    middle_portion = y[(y > lower_bound) & (y < upper_bound)]
    
    # Mean of the middle (100-2*p)% of the data
    mean_middle = np.mean(middle_portion)
    
    # Std of the middle (100-2*p)% of the data
    std_middle = np.std(middle_portion, ddof=1) / np.std(y, ddof=1)  # [although std(y) should be 1]
    
    if justMe is None:
        return {'mean': mean_middle, 'std': std_middle}
    elif justMe == 'mean':
        return mean_middle
    elif justMe == 'std':
        return std_middle
    else:
        raise ValueError(f"Unknown option '{justMe}'")
