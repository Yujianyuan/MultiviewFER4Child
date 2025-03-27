import numpy as np
from numba import jit, prange

# construct multiscale series
def custom_granulate_time_series(time_series, scale):
    """
    The time series is coarsened according to a given scale factor 
    Nans are returned if more than 50% of Nans are included
    """
    n = len(time_series)
    granulated_time_series = []
    for i in range(0, n, scale):
        window = time_series[i:i + scale]
        if np.count_nonzero(~np.isnan(window)) / scale >= 0.5:
            granulated_time_series.append(np.nanmean(window))
        else:
            granulated_time_series.append(np.nan)
    return granulated_time_series

# compute sample entropy
@jit(nopython=True, parallel=True)
def sample_entropy(time_series, m, r=None):
    """
    Calculate sample entropy of a time series with embedding dimension m and tolerance r.
    This function can handle NaNs. If a vector comparison involves NaNs, that pair is skipped.

    :param time_series: 1D array-like, the time series data.
    :param m: embedding dimension
    :param r: tolerance (if None, defaults to 0.1*std of the non-NaN elements)
    :return: SampEn(m, r)
    """
    ts = np.array(time_series, dtype=float)

    if r is None:
        r = 0.1 * np.nanstd(ts)
        if np.isnan(r) or r == 0:
            r = 1e-10

    N = len(ts)
    if N < 2*m:
        return np.nan  

    X_m = np.array([ts[i:i+m] for i in range(N-m+1)])
    X_m1 = np.array([ts[i:i+m+1] for i in range(N-m)])  # For m+1 dimension

    # Function to count matches for dimension m
    def count_matches(X, threshold):
        """
        Count the number of pairs (i,j), i<j, of vectors in X that match within threshold.
        If either vector in a pair contains NaN, that pair is skipped.
        """
        count = 0
        length = len(X)
        for i in range(length - 1):
            Xi = X[i]
            if np.isnan(Xi).any():
                continue
            for j in range(i + 1, length):
                Xj = X[j]
                if np.isnan(Xj).any():
                    continue
                dist = np.max(np.abs(Xi - Xj))
                if dist < threshold:
                    count += 1
        return count

    B_m = count_matches(X_m, r)
    A_m = count_matches(X_m1, r)

    if B_m == 0 or A_m == 0:
        return np.nan
    sampen = -np.log(A_m / B_m)
    return sampen

def multiscale_entropy(time_series, max_scale, m, r):
    """
    Calculate multiscale entropy
    :param time_series: Input time series (one-dimensional)
    :param max_scale: Maximum scale factor
    :param m: Embedding dimension
    :param r: Similarity threshold
    :return: List of sample entropy for each scale
    """
    mse = []
    for scale in range(1, max_scale + 1):
        coarse_grained_ts = custom_granulate_time_series(time_series, scale)

        se = sample_entropy(coarse_grained_ts, m, r)
        mse.append(se)  

    return mse



def get_dMSE(sub_emot_dict, m, max_scale, r_rate, sigma):
    '''
    sub_emot_dict -- key: frameID  
                     value: 1 dim facial expression dynamics of frameID
    '''
    
    valid_data = np.array([v for v in sub_emot_dict.values()]) 

    r = r_rate * sigma  
    mse_results = multiscale_entropy(valid_data, max_scale, m, r)

    return mse_results

