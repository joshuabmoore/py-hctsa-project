import numpy as np

def EN_wentropy(y, whaten = 'shannon', p = None):
    """
    Entropy of time series using wavelets.
    Uses a python port of the MATLAB wavelet toolbox wentropy function.

    Parameters:
    ----------
    y : array_like
        Input time series
    whaten : str, optional
        The entropy type:
        - 'shannon' (default)
        - 'logenergy'
        - 'threshold' (with a given threshold)
        - 'sure' (with a given parameter)
        (see the wentropy documentation for information)
    p : any, optional
        the additional parameter needed for threshold and sure entropies

    Returns:
    --------
    out : float
        Entropy value. 
    """
    N = len(y)

    if whaten == 'shannon':
        # compute Shannon entropy
        out = wentropy(y, 'shannon')/N
    elif whaten == 'logenergy':
        out = wentropy(y, 'logenergy')/N
    elif whaten == 'threshold':
        # check that p has been provided
        if p is not None:
            out = wentropy(y, 'threshold', p)/N
        else:
            raise ValueError("threshold requires an additional parameter, p.")
    elif whaten == 'sure':
        if p is not None:
            out = wentropy(y, 'sure', p)/N
        else:
            raise ValueError("sure requires an additional parameter, p.")
    else:
        raise ValueError(f"Unknown entropy type {whaten}")

    return out

# helper functions
# taken from https://github.com/fairscape/hctsa-py/blob/master/PeripheryFunctions/wentropy.py
def wentropy(x, entType = 'shannon', additionalParameter = None):

    if entType == 'shannon':
        x = np.power(x[ x != 0 ],2)
        return - np.sum(np.multiply(x,np.log(x)))

    elif entType == 'threshold':
        if additionalParameter is None or isinstance(additionalParameter, str):
            return None
        x = np.absolute(x)
        return np.sum((x > additionalParameter))

    elif entType == 'norm':
        if additionalParameter is None or isinstance(additionalParameter,str) or additionalParameter < 1:
            return None
        x = np.absolute(x)
        return np.sum(np.power(x, additionalParameter))

    elif entType == 'sure':
        if additionalParameter is None or isinstance(additionalParameter,str):
            return None

        N = len(x)
        x2 = np.square(x)
        t2 = additionalParameter**2
        xgt = np.sum((x2 > t2))
        xlt = N - xgt

        return N - (2*xlt) + (t2 *xgt) + np.sum(np.multiply(x2,(x2 <= t2)))

    elif entType == 'logenergy':
        x = np.square(x[x != 0])
        return np.sum(np.log(x))

    else:
        print("invalid entropy type")
        return None
    