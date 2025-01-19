import os
import json
import pandas as pd
import numpy as np
import math
import csv
from scipy.spatial.distance import jensenshannon
from numba import jit, prange
from sigma import *

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
        # Check if each vector in the window has any dimension as NaN, then determine if the entire vector is NaN
        if np.any(np.isnan(window), axis=1).sum() / scale >= 0.5:
            granulated_time_series.append([np.nan] * 7)
        else:
            # Check if there is at least one vector that has a non-NaN value in all dimensions
            if np.all(np.isnan(window), axis=0).any():
                # If any dimension is completely NaN, set the result to be a NaN vector
                granulated_time_series.append([np.nan] * 7)
            else:
                # Calculate the mean for each dimension, ignoring NaNs
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

    # Determine r if not given
    if r is None:
        r = 0.1 * np.nanstd(ts)
        if np.isnan(r) or r == 0:
            r = 1e-10

    if N < 2*m:
        # Not enough data
        return np.nan

    # Construct embedding vectors
    # X_m: shape (N-m+1, m, D)
    X_m = np.array([ts[i:i+m] for i in range(N - m + 1)])
    # X_m1: shape (N-m, m+1, D)
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
            # Check if Xi contains NaN
            if np.isnan(Xi).any():
                # No valid pairs with Xi
                continue
            for j in range(i + 1, length):
                Xj = X[j]
                # Check if Xj contains NaN
                if np.isnan(Xj).any():
                    continue
                # Compute distance: for each time step in embedding, calculate JS distance, then take max
                dist_vals = []
                # Xi and Xj are shape (m or m+1, D)
                for k in range(Xi.shape[0]):
                    # Extract distributions (1D arrays)
                    dist_val = jensenshannon(Xi[k], Xj[k])
                    # jensenshannon returns a scalar distance
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
        # Construct coarse-grained time series
        coarse_grained_ts = custom_granulate_time_series_multidim(time_series, scale)
        # Calculate sample entropy, can handle NaNs in the input
        se = sample_entropy_multidim(coarse_grained_ts, m, r)
        mse.append(se)  # Handle the possibility of an empty list

    return mse


def get_multiscale_entropy_multidim(emot_dict, m=2, max_scale=10, r_rate=0.15, sigma = -1000):
    # Extract data from dictionary without removing NaNs
    valid_data = np.array([v for v in emot_dict.values()]) 
    valid_data_for_std = np.array([v for v in emot_dict.values() if not np.all(np.isnan(v))])

    r = r_rate*sigma # Similarity threshold
    mse_results = multiscale_entropy_multidim(valid_data, max_scale, m, r)
    return mse_results


def to_four_decimals(input_list):
    return [round(element, 4) for element in input_list]

def get_real_act(full_act):
    #some activity name may include [], like [a1]
    if '[' in full_act:
        full_act = full_act.split('[')[0]
    return full_act    


def PMSE(info_base, avg_sgm_dict, all_emot):
    feat_dict = {}

    id_list = os.listdir(info_base)
    for id_n in id_list:
        if id_n not in feat_dict.keys():
            feat_dict[id_n] = {}

        id_path = os.path.join(info_base, id_n)
        csv_list = os.listdir(id_path)
        for csv_name in csv_list:
            if 'csv' not in csv_name:
                continue

            csv_path = os.path.join(id_path, csv_name)
            activity = csv_name.split('.')[0].split('_')[1][1:-1]

            if activity not in feat_dict[id_n].keys():
                feat_dict[id_n][activity] = {}
            else:
                continue

            kid_dict = {}
            df = pd.read_csv(csv_path)
            for index, row in df.iterrows():
                frameID = row['frameID']
                personID = row['personID']

                # Only compute the emotion of the children
                if personID == 'kid':
                    emot_list = []
                    for item in all_emot:
                        emot_list.append(row[item])
                    kid_dict[frameID] = emot_list

            # Choose kid dict
            chosen_dict = kid_dict
            real_act = get_real_act(activity)

            # compute PMSE
            mse_results = get_multiscale_entropy_multidim(
                chosen_dict, r_rate=0.15, sigma = avg_sgm_dict[real_act][0]
            )

            # Compute entropy for each scale
            scale_num = len(mse_results)
            for idx in range(scale_num):
                feat_dict[id_n][activity][
                    'js_mse(avg_sgm,s=%d,r=0.15)'%(idx+1)
                ]= to_four_decimals([mse_results[idx]])

            print('finish:', id_n, activity)

    return feat_dict

