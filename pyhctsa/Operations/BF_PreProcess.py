import numpy as np
from BF_makeBuffer import BF_MakeBuffer
from CO import FirstCrossing

def BF_PreProcess(y, preProcessHow = None):
    """
    Preprocess a time series, y.

    Parameters:
    -----------
    y (array-like):
        the input time series
    preProcessHow (str, optional):
        how to pre-process the data
        - 'diff1'
        - 'rescale_tau'
    """

    if preProcessHow is not None:
        if preProcessHow == 'diff1':
            y = np.diff(y)
        elif preProcessHow == 'rescale_tau':
            tau = FirstCrossing(y, 'ac', 0, 'discrete')
            y_buffer = BF_MakeBuffer(y, tau)
            y = np.mean(y_buffer, 1)
        else:
            raise ValueError(f"Unknown preprocessing setting: {preProcessHow}")    
    return y
