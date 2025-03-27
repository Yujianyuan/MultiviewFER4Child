import numpy as np
from scipy.spatial.distance import jensenshannon
from numba import jit, prange

# construct multi-dimention multiscale series
def custom_granulate_time_series_multidim(time_series, scale):
    """
    Coarsen the time series according to the given scale factor. Each element is a 7-dimensional vector.
    If more than 50% of the vectors in a window have at least one dimension as NaN, the result for that window will be a NaN vector.
    Otherwise, calculate the mean for each dimension, ignoring NaNs.
    """
    n = len(time_series)
    granulated_time_series = []
    for i in range(0, n, scale):
        window = time_series[i:i + scale]
        if np.any(np.isnan(window), axis=1).sum() / scale >= 0.5:
            granulated_time_series.append([np.nan] * 7)
        else:
            if np.all(np.isnan(window), axis=0).any():
                granulated_time_series.append([np.nan] * 7)
            else:
                mean_values = np.nanmean(window, axis=0)
                granulated_time_series.append(mean_values)
    return granulated_time_series

@jit(nopython=True, parallel=True)
def sample_entropy_multidim(time_series, m, r=None):
    """
    Calculate sample entropy for a multi-dimensional time series.
    :param time_series: 2D array-like of shape (N, D), N is length, D is feature dimension.
                        Each row is a time point with D-dimensional feature vector.
    :param m: Embedding dimension
    :param r: Tolerance. If None, defaults to 0.1 * nanstd of the data.
    :return: Sample Entropy value (float) or np.nan if not definable.
    """
    ts = np.array(time_series, dtype=float)

    N, D = ts.shape

    if r is None:
        r = 0.1 * np.nanstd(ts)
        if np.isnan(r) or r == 0:
            r = 1e-10

    if N < 2*m:
        return np.nan

    X_m = np.array([ts[i:i+m] for i in range(N - m + 1)])
    X_m1 = np.array([ts[i:i+m+1] for i in range(N - m)])

    def count_matches(X, threshold):
        """
        Count the number of matches pairs (i < j) of embedding vectors in X.
        Each embedding vector: shape (m_or_m+1, D)
        Distance measure: Jensen-Shannon distance per step, then take max over the embedding window.
        Skip pairs that involve NaNs.
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
                dist_vals = []
                for k in range(Xi.shape[0]):
                    dist_val = jensenshannon(Xi[k], Xj[k])
                    dist_vals.append(dist_val)
                dist = np.max(dist_vals)
                if dist < threshold:
                    count += 1
        return count

    B_m = count_matches(X_m, r)
    A_m = count_matches(X_m1, r)

    if B_m == 0 or A_m == 0:
        return np.nan
    sampen = -np.log(A_m / B_m)
    return sampen




def multiscale_entropy_multidim(time_series, max_scale, m, r):
    """
    Calculate multiscale entropy.
    :param time_series: Input time series (1-dimensional)
    :param max_scale: Maximum scale factor
    :param m: Embedding dimension
    :param r: Similarity threshold
    :return: List of sample entropies for each scale
    """
    mse = []
    for scale in range(1, max_scale + 1):
        coarse_grained_ts = custom_granulate_time_series_multidim(time_series, scale)
        se = sample_entropy_multidim(coarse_grained_ts, m, r)
        mse.append(se)  
    return mse


def get_pMSE(emot_dict, m, max_scale, r_rate, sigma = -1000):
    '''
    emot_dict -- key: frameID  
                 value: 7 dim facial expression probability of frameID
    '''
    
    valid_data = np.array([v for v in emot_dict.values()]) 

    r = r_rate*sigma 
    mse_results = multiscale_entropy_multidim(valid_data, max_scale, m, r)
    return mse_results

