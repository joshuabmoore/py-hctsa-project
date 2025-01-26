import numpy as np

def CO_TranslateShape(y, shape = 'circle', d = 2, howToMove = 'pts'):
    """
    """
    N = len(y)

    # add a time index
    ty = [np.arange(N).T, y] # has increasing integers as time in the first column
    #-------------------------------------------------------------------------------
    # Generate the statistics on the number of points inside the shape as it is
    # translated across the time series
    #-------------------------------------------------------------------------------
    

