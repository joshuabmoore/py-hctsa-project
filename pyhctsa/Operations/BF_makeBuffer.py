import numpy as np


def BF_MakeBuffer(y, bufferSize):
    """
    Make a buffered version of a time series.

    Parameters
    ----------
    y : array-like
        The input time series.
    bufferSize : int
        The length of each buffer segment.

    Returns
    -------
    y_buffer : ndarray
        2D array where each row is a segment of length `bufferSize` 
        corresponding to consecutive, non-overlapping segments of the input time series.
    """
    N = len(y)

    numBuffers = int(np.floor(N/bufferSize))

    # may need trimming
    y_buffer = y[:numBuffers*bufferSize]
    # then reshape
    y_buffer = y_buffer.reshape((numBuffers,bufferSize))

    return y_buffer
