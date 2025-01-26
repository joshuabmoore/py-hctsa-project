from scipy import stats

def DN_Cumulants(y, cumWhatMay = 'skew1'):
    """
    Distributional moments of the input data.
    Very simple function that uses the skewness and kurtosis functions
    to calculate these higher order moments of input time series, y

    Parameters:
    ----------
    y (array-like) : the input time series
    cumWhatMay (str, optional) : the type of higher order moment
        (i) 'skew1', skewness
        (ii) 'skew2', skewness correcting for bias
        (iii) 'kurt1', kurtosis
        (iv) 'kurt2', kurtosis correcting for bias

    Returns:
    --------
    float : the higher order moment.
    """
    if cumWhatMay == 'skew1':
        out = stats.skew(y)
    elif cumWhatMay == 'skew2':
        out = stats.skew(y, bias=False)
    elif cumWhatMay == 'kurt1':
        out = stats.kurtosis(y, fisher=False)
    elif cumWhatMay == 'kurt2':
        out = stats.kurtosis(y, bias=False, fisher=False)
    else:
        raise ValueError('Requested Unknown cumulant must be: skew1, skew2, kurt1, or kurt2')
    
    return out
