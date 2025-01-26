import numpy as np
from Operations.CO_FirstMin import CO_FirstMin
from Operations.CO_FirstCrossing import CO_FirstCrossing

def BF_Embed(y, tau = None, m = None, makeSignal = None, randomSeed = None, beVocal = False):
    """
    """
     # (1) Time-delay, tau
    if tau is None:
        tau = 1 # default time delay is 1
    else:
        if isinstance(tau, 'str'):
            if tau == 'mi':
                # first min of MI function
                tau = CO_FirstMin(y, 'mi')
                if np.isnan(tau):
                    raise ValueError("Could not get time delay by mutual information (time series too short?)")
            elif tau == 'ac': # first zero crossing of the ACF
                tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
                if np.isnan(tau):
                    raise ValueError("Could not get time delay by ACF (time series too short?)")
            else:
                raise ValueError(f"Invalid time-delay method {tau}")
        else:
            raise ValueError(f"Invalid time-delay method {tau}. Either supply an integer tau, or specify 'mi' or 'ac'.")

    # Determine the embedding dimension, m
    if m is None:
        m = 2 # embed in 2 dimensional space by default
    else:
        # use a routine to inform m
        if isinstance(m, list):
            if m[0] == 'fnnsmall':
                # Use Michael Small's fnn code
                pass 







