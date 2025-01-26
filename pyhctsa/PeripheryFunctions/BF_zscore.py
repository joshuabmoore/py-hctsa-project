import numpy as np

def BF_zscore(input_data):
    """
    Z-score the input data vector.

    Parameters:
    input_data (array-like): The input time series (or any vector).

    Returns:
    numpy.ndarray: The z-scored transformation of the input.

    Raises:
    ValueError: If input_data contains NaN values.
    """
    # Convert input to numpy array
    input_data = np.array(input_data)

    # Check for NaNs
    if np.isnan(input_data).any():
        raise ValueError('input_data contains NaNs')

    # Z-score twice to reduce numerical error
    zscored_data = (input_data - np.mean(input_data)) / np.std(input_data)
    zscored_data = (zscored_data - np.mean(zscored_data)) / np.std(zscored_data)

    return zscored_data
