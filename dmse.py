import os
import json
import pandas as pd
import numpy as np
import math
import csv
from sigma import *

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
def sample_entropy(time_series, sample_length, tolerance=None):
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if tolerance is None:
        # Excluding NaN values for standard deviation calculation
        tolerance = 0.1 * np.nanstd(time_series)

    sample_length = sample_length - 1
    n = len(time_series)
    N_temp = np.zeros(sample_length + 2)
    N_temp[0] = n * (n - 1) / 2

    for i in range(n - sample_length - 1):
        template = time_series[i : (i + sample_length + 1)]

        # Skip if NaN is in the template
        if np.isnan(template).any():
            continue

        rem_time_series = time_series[i + 1 :]
        search_list = np.arange(len(rem_time_series) - sample_length, dtype=np.int32)
        
        for length in range(1, len(template) + 1):
            # Skip comparison if NaN is in the elements to compare
            hit_list = np.logical_and(
                np.abs(rem_time_series[search_list] - template[length - 1]) < tolerance,
                ~np.isnan(rem_time_series[search_list])
            )
            N_temp[length] += np.sum(hit_list)
            search_list = search_list[hit_list] + 1

    # Avoid division by zero in case no matches are found
    N_temp[N_temp == 0] = np.nan
    sampen = -np.log(N_temp[1:] / N_temp[:-1])
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
        mse.append(se[0] if len(se) > 0 else np.nan)  # Handle the possibility of an empty list

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

