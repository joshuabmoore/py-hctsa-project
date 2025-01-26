import numpy as np

def DN_MinMax(y, minOrMax):
    """
    The maximum and minimum values of the input data vector.

    """
    if minOrMax == 'max':
        out = max(y)
    elif minOrMax == 'min':
        out = min(y)
    else:
        raise ValueError(f"Unknown method '{minOrMax}'")
    
    return out
