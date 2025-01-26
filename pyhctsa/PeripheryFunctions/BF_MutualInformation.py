import numpy as np

def BF_MutualInformation(v1, v2, r1 = 'range', r2 = 'range', numBins = 10):
    """
    Compute mutual information between two data vectors using bin counting.

    Parameters:
    -----------
        v1 (array-like): The first input vector
        v2 (array-like): The second input vector
        r1 (str or list): The bin-partitioning method for v1 ('range', 'quantile', or [min, max])
        r2 (str or list): The bin-partitioning method for v2 ('range', 'quantile', or [min, max])
        numBins (int): The number of bins to partition each vector into (default : 10)

    Returns:
    --------
        float: The mutual information computed between v1 and v2
    """
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()

    if len(v1) != len(v2):
        raise ValueError("Input vectors must be the same length")

    N = len(v1)

    # Create histograms
    edges_i = SUB_GiveMeEdges(r1, v1, numBins)
    edges_j = SUB_GiveMeEdges(r2, v2, numBins)

    ni, _ = np.histogram(v1, edges_i)
    nj, _ = np.histogram(v2, edges_j)

    # Create a joint histogram
    hist_xy, _, _ = np.histogram2d(v1, v2, [edges_i, edges_j])

    # Normalize counts to probabilities
    p_i = ni[:numBins] / N
    p_j = nj[:numBins] / N
    p_ij = hist_xy / N
    p_ixp_j = np.outer(p_i, p_j)

    # Calculate mutual information
    mask = (p_ixp_j > 0) & (p_ij > 0)
    if np.any(mask):
        mi = np.sum(p_ij[mask] * np.log(p_ij[mask] / p_ixp_j[mask]))
    else:
        print("The histograms aren't catching any points. Perhaps due to an inappropriate custom range for binning the data.")
        mi = np.nan

    return mi

def SUB_GiveMeEdges(r, v, nbins):
    EE = 1E-6 # this small addition gets lost in the last bin
    if r == 'range':
            return np.linspace(np.min(v), np.max(v) + EE, nbins + 1)
    elif r == 'quantile': # bin edges based on quantiles
        edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
        edges[-1] += EE
        return edges
    elif isinstance(r, (list, np.ndarray)) and len(r) == 2: # a two-component vector
        return np.linspace(r[0], r[1] + EE, nbins + 1)
    else:
        raise ValueError(f"Unknown partitioning method '{r}'")
