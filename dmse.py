import os
import json
import pandas as pd
import numpy as np
import math
import csv
from sigma import *
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

    # If no tolerance specified, use 0.1 * nanstd
    if r is None:
        r = 0.1 * np.nanstd(ts)
        if np.isnan(r) or r == 0:
            # If std is zero or NaN, use a small epsilon to avoid division by zero
            r = 1e-10

    N = len(ts)
    if N < 2*m:
        return np.nan  # Not enough data to form m and m+1 dimension vectors

    # Construct m-dim and (m+1)-dim embeddings
    # X_m[i] = (x[i], x[i+1], ..., x[i+m-1])
    # There are (N-m+1) such m-dimensional vectors
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
            # If Xi has NaN, pairs involving Xi are only valid if Xj has no NaN
            # Actually, if Xi has NaN, no match can be formed with it since we can't define distance properly.
            # But let's handle it explicitly:
            if np.isnan(Xi).any():
                # All pairs with Xi are invalid
                continue
            for j in range(i + 1, length):
                Xj = X[j]
                # If Xj has NaN, skip
                if np.isnan(Xj).any():
                    continue
                # Check max distance between Xi and Xj
                dist = np.max(np.abs(Xi - Xj))
                if dist < threshold:
                    count += 1
        return count

    B_m = count_matches(X_m, r)
    A_m = count_matches(X_m1, r)

    # According to SampEn definition:
    # B(m) = B_m / (N-m)*(N-m-1)/2 is often conceptual, but since B_m counts all pairs i<j, it's already the count of pairs.
    # Similarly for A_m.
    # SampEn = -ln( A_m / B_m )
    # Need to handle division by zero or no matches cases
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
        # Construct coarse-grained time series
        coarse_grained_ts = custom_granulate_time_series(time_series, scale)

        # Calculate sample entropy, able to handle input containing NaNs
        se = sample_entropy(coarse_grained_ts, m, r)
        mse.append(se)  

    return mse

def get_multiscale_entropy(emot_dict, m=2, max_scale=10, r_rate=0.15, sigma = 0.2):
    # Extract data from dictionary without removing NaNs
    valid_data = np.array([v for v in emot_dict.values()]) 
    valid_data_for_std = np.array([v for v in emot_dict.values() if not np.all(np.isnan(v))])

    r = r_rate * sigma  # Similarity threshold
    mse_results = multiscale_entropy(valid_data, max_scale, m, r)

    return mse_results


def to_four_decimals(input_list):
    return [round(element, 4) for element in input_list]

def get_real_act(full_act):
    #some activity name may include [], like [a1]
    if '[' in full_act:
        full_act = full_act.split('[')[0]
    return full_act    


def DMSE(info_base, avg_sgm_dict, all_emot):
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

            # Get sub sequence
            emot_sub_seq = get_sub_seq(chosen_dict)

            # Average
            mse_results = get_multiscale_entropy(
                emot_sub_seq, sigma=avg_sgm_dict[real_act][0]
            )

            # Compute entropy for each scale
            scale_num = len(mse_results)
            for idx in range(scale_num):
                feat_dict[id_n][activity][
                    'sub_mse(avg_sgm,t=%d,delta=%d,s=%d)' % (1, 0, idx + 1)
                ] = to_four_decimals([mse_results[idx]])

            print('finish:', id_n, activity)

    return feat_dict

