def DN_ProportionValues(x, propWhat='positive'):
    """
    Proportion of values in a data vector.
    Returns statistics on the values of the data vector: the proportion of zeros,
    the proportion of positive values, and the proportion of values greater than or
    equal to zero.

    Parameters:
    x (array-like): the input time series
    propWhat (str): the proportion of a given type of value in the time series: 
        (i) 'zeros': values that equal zero
        (ii) 'positive': values that are strictly positive
        (iii) 'geq0': values that are greater than or equal to zero

    Returns:
    out (float) : proportion of given type of value
    """

    N = len(x)

    if propWhat == 'zeros':
        # returns the proportion of zeros in the input vector
        out = sum(x == 0) / N
    elif propWhat == 'positive':
        out = sum(x > 0) / N
    elif propWhat == 'geq0':
        out = sum(x >= 0) / N
    else:
        raise ValueError(f"Unknown condition to measure: {propWhat}")

    return out
