import numpy as np
from Operations.CO_FirstCrossing import CO_FirstCrossing

def SB_CoarseGrain(y, howtocg, numGroups):
    """
    Coarse-grains a continuous time series to a discrete alphabet.

    Parameters:
    -----------
    y : array-like
        The input time series.
    howtocg : str
        The method of coarse-graining.
        Options: 'updown', 'quantile', 'embed2quadrants', 'embed2octants'
    numGroups : int
        Specifies the size of the alphabet for 'quantile' and 'updown',
        or sets the time delay for the embedding subroutines.

    Returns:
    --------
    yth : array-like
        The coarse-grained time series.
    """
    y = np.array(y)
    N = len(y)

    if howtocg not in ['updown', 'quantile', 'embed2quadrants', 'embed2octants']:
        raise ValueError(f"Unknown coarse-graining method '{howtocg}'")

    # Some coarse-graining/symbolization methods require initial processing:
    if howtocg == 'updown':
        y = np.diff(y)
        N = N - 1 # the time series is one value shorter than the input because of differencing
        howtocg = 'quantile' # successive differences and then quantiles

    elif howtocg in ['embed2quadrants', 'embed2octants']:
        # Construct the embedding
        if numGroups == 'tau':
            # First zero-crossing of the ACF
            tau = CO_FirstCrossing(y, 'ac', 0, 'discrete')
        else:
            tau = numGroups
        
        if tau > N/25:
            tau = N // 25

        m1 = y[:-tau]
        m2 = y[tau:]

        # Look at which points are in which angular 'quadrant'
        upr = m2 >= 0 # points above the axis
        downr = m2 < 0 # points below the axis 

        q1r = np.logical_and(upr, m1 >= 0) # points in quadrant 1
        q2r = np.logical_and(upr, m1 < 0) # points in quadrant 2
        q3r = np.logical_and(downr, m1 < 0) # points in quadrant 3
        q4r = np.logical_and(downr, m1 >= 0) # points in quadrant 4
    
    # Do the coarse graining
    if howtocg == 'quantile':
        th = np.quantile(y, np.linspace(0, 1, numGroups + 1), method='hazen') # thresholds for dividing the time-series values
        th[0] -= 1  # Ensure the first point is included
        # turn the time series into a set of numbers from 1:numGroups
        yth = np.zeros(N, dtype=int)
        for i in range(numGroups):
            yth[(y > th[i]) & (y <= th[i+1])] = i + 1

    elif howtocg == 'embed2quadrants': # divides based on quadrants in a 2-D embedding space
        # create alphabet in quadrants -- {1,2,3,4}
        yth = np.zeros(len(m1), dtype=int)
        yth[q1r] = 1
        yth[q2r] = 2
        yth[q3r] = 3
        yth[q4r] = 4
        
    elif howtocg == 'embed2octants': # divide based on octants in 2-D embedding space
        o1r = np.logical_and(q1r, m2 < m1) # points in octant 1
        o2r = np.logical_and(q1r, m2 >= m1) # points in octant 2
        o3r = np.logical_and(q2r, m2 >= -m1) # points in octant 3
        o4r = np.logical_and(q2r, m2 < -m1) # points in octant 4
        o5r = np.logical_and(q3r, m2 >= m1) # points in octant 5
        o6r = np.logical_and(q3r, m2 < m1) # points in octant 6
        o7r = np.logical_and(q4r, m2 < -m1) # points in octant 7
        o8r = np.logical_and(q4r, m2 >= -m1) # points in octant 8

        # create alphabet in octants -- {1,2,3,4,5,6,7,8}
        yth = np.zeros(len(m1), dtype=int)
        yth[o1r] = 1
        yth[o2r] = 2
        yth[o3r] = 3
        yth[o4r] = 4
        yth[o5r] = 5
        yth[o6r] = 6
        yth[o7r] = 7
        yth[o8r] = 8

    if np.any(yth == 0):
        raise ValueError('All values in the sequence were not assigned to a group')

    return yth 
