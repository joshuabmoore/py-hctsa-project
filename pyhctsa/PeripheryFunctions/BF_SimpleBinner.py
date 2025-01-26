import numpy as np

def BF_SimpleBinner(xData, numBins):
    """
    Generate a histogram from equally spaced bins.
   
    Parameters:
    xData (array-like): A data vector.
    numBins (int): The number of bins.

    Returns:
    tuple: (N, binEdges)
        N (numpy.ndarray): The counts
        binEdges (numpy.ndarray): The extremities of the bins.
    """
    minX = np.min(xData)
    maxX = np.max(xData)
    
    # Linearly spaced bins:
    binEdges = np.linspace(minX, maxX, numBins + 1)
    N = np.zeros(numBins, dtype=int)
    
    for i in range(numBins):
        if i < numBins - 1:
            N[i] = np.sum((xData >= binEdges[i]) & (xData < binEdges[i+1]))
        else:
            # the final bin
            N[i] = np.sum((xData >= binEdges[i]) & (xData <= binEdges[i+1]))
    
    return N, binEdges
