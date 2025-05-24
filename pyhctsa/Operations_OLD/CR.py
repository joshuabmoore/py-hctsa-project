import numpy as np
from typing import Union
from CO import FirstCrossing

def RAD(x : list, tau : Union[int, str] = 1, centre : bool = True):
    """
    Harris, Gollo, & Fulcher's rescaled auto-density (RAD) noise-insensitive
    metric for inferring the distance to criticality.

    Parameters
    ----------
    x: array
        A time-series input vector

    centre : boolean
        Whether to centre the time series and take absolute values

    tau: integer
        The embedding and differencing delay in units of the timestep

    Returns
    -------
    The RAD feature value.
    """

    # ensure that x is in the form of a numpy array
    x = np.array(x)
    
    # if specified: centre the time series and take the absolute value
    if centre:
        x = x - np.median(x)
        x = np.abs(x)
    
    # if specified, make tau the first crossing of the AC function
    if isinstance(tau, str):
        if tau == "tau":
            tau = FirstCrossing(x, 'ac', 0, 'discrete')
        else:
            raise ValueError(f"Unknown operation {tau}")

    # Delay embed at interval tau
    y = x[tau:]
    x = x[:-tau]

    # Median split
    subMedians = x < np.median(x)
    superMedianSD = np.std(x[~subMedians], ddof=1)
    subMedianSD = np.std(x[subMedians], ddof=1)

    # Properties of the auto-density
    sigma_dx = np.std(y - x, ddof=1)
    densityDifference = (1/superMedianSD) - (1/subMedianSD)

    # return RAD
    return sigma_dx * densityDifference
