import os
import json
import pandas as pd
import numpy as np
import math
import csv
from scipy.spatial.distance import jensenshannon
# from numba import jit, prange
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


def sample_entropy_multidim(time_series, sample_length, tolerance=None):
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if tolerance is None:
        # We calculate a default tolerance based on JS divergence if not provided
        # This is a placeholder, should be defined based on domain knowledge or data analysis
        tolerance = 0.1*compute_js_nanstd(time_series)

    sample_length = sample_length - 1
    n = len(time_series)
    N_temp = np.zeros(sample_length + 2)
    N_temp[0] = n * (n - 1) / 2

    for i in range(n - sample_length - 1):
        template = time_series[i: (i + sample_length + 1)]

        # Skip if NaN is in the template
        if np.isnan(template).any():
            continue

        rem_time_series = time_series[i + 1:]
        search_list = np.arange(len(rem_time_series) - sample_length, dtype=np.int32)

        for length in range(1, len(template) + 1):
            valid_indices = [j for j in search_list if not np.isnan(rem_time_series[j]).any()]
            js_distances = np.array([jensenshannon(template[length - 1], rem_time_series[j]) for j in valid_indices])
            
            hit_list = js_distances < tolerance
            valid_hits = np.array(valid_indices)[hit_list]  # Apply hit_list to valid_indices

            N_temp[length] += len(valid_hits)
            
            # Update search_list only with indices that had valid hits
            search_list = valid_hits + 1

    # Avoid division by zero in case no matches are found
    N_temp[N_temp == 0] = np.nan
    sampen = -np.log(N_temp[1:] / N_temp[:-1])
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
        mse.append(se[0] if len(se) > 0 else np.nan)  # Handle the possibility of an empty list

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

