import numpy as np
from Operations_OLD.SY_SlidingWindow import SY_SlidingWindow

def CO_TranslateShape(y, shape = 'circle', d = 2, howToMove = 'pts'):
    """
    """
    y = np.array(y, dtype=float)
    N = len(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.shape[1] > y.shape[0]:
        y = y.T

    # add a time index
    ty = np.column_stack((np.arange(1, N+1), y[:, 0])) # has increasing integers as time in the first column
    #-------------------------------------------------------------------------------
    # Generate the statistics on the number of points inside the shape as it is
    # translated across the time series
    #-------------------------------------------------------------------------------
    if howToMove == 'pts':

        if shape == 'circle':

            r = d # set radius
            w = int(np.floor(r))
            rnge = np.arange(1 + w, N - w + 1)
            NN = len(rnge) # number of admissible points
            np_counts = np.zeros(NN, dtype=int)

            for i in range(NN):
                idx = rnge[i]
                start = idx - w - 1
                end = idx + w
                win = ty[start:end, :]
                difwin = win - ty[idx - 1, :]
                squared_dists = np.sum(difwin**2, axis=1)
                np_counts[i] = np.sum(squared_dists <= r**2)

        elif shape == 'rectangle':

            w = d
            rnge = np.arange(1 + w, N - w + 1)
            NN = len(rnge)
            np_counts = np.zeros(NN, dtype=int)

            for i in range(NN):
                idx = rnge[i]
                start = (idx - w) - 1
                end = (idx + w)
                np_counts[i] = np.sum(
                    np.abs(y[start:end, 0]) <= np.abs(y[i, 0])
                )
        else:
            raise ValueError(f"Unknown shape {shape}. Choose either 'circle' or 'rectangle'")
    else:
        raise ValueError(f"Unknown setting for 'howToMove' input: '{howToMove}'. Only option is currently 'pts'.")

    # compute stats on number of hits inside the shape
    out = {}
    out["max"] = np.max(np_counts)
    out["std"] = np.std(np_counts, ddof=1)
    out["mean"] = np.mean(np_counts)
    
    # count the hits
    vals, hits = np.unique_counts(np_counts)
    max_val = np.argmax(hits)
    out["npatmode"] = hits[max_val]/NN
    out["mode"] = vals[max_val]

    count_types = ["ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens", "elevens"]
    for i in range(1, 12):
        if 2*w + 1 >= i:
            out[f"{count_types[i-1]}"] = np.mean(np_counts == i)
    
    out['statav2_m'] = SY_SlidingWindow(np_counts, 'mean', 'std', 2, 1)
    out['statav2_s'] = SY_SlidingWindow(np_counts, 'std', 'std', 2, 1)
    out['statav3_m'] = SY_SlidingWindow(np_counts, 'mean', 'std', 3, 1)
    out['statav3_s'] = SY_SlidingWindow(np_counts, 'std', 'std', 3, 1)
    out['statav4_m'] = SY_SlidingWindow(np_counts, 'mean', 'std', 4, 1)
    out['statav4_s'] = SY_SlidingWindow(np_counts, 'std', 'std', 4, 1)

    return out
