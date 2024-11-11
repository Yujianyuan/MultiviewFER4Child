import os
import json
import pandas as pd
import numpy as np
import math
import csv
import re
import ast


# base_dict = json.load(open('./feature_dict/base_koala.json','r'))
# sub_mse_param_sgm_dict = json.load(open('./feature_dict/dmse_koala.json','r'))
# jsmse_avgsgm_r15_dict = json.load(open('./feature_dict/pmse_koala.json','r'))

def write_to_csv(base_dict, sub_mse_param_sgm_dict, jsmse_avgsgm_r15_dict, out_path):
    # out_path = './feature_csv/base_dmse_jsmse_hasnan_koala.csv'
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        head_list = ['ID', 'activity', 'mean']#, 'std', 'entropy']

        for (t,delta) in [(1,0)]:#avg
            for scale in range(1,10+1):
                head_list += ['sub_mse(avg_sgm,t=%d,delta=%d,s=%d)'%(t,delta,scale)]
        
        for scale in range(1,5+1):
            head_list += ['js_mse(avg_sgm,s=%d,r=0.15)'%(scale)]

        writer.writerow(head_list)

        for id_n in base_dict.keys():
            for act in base_dict[id_n].keys():
                if act == 'RJAcar[a87]':
                    continue

                row_list = [id_n,act]
                for base_metric in base_dict[id_n][act].keys():
                    row_list.append(base_dict[id_n][act][base_metric])

                for sub_mse_param_sgm_metric in sub_mse_param_sgm_dict[id_n][act].keys():
                    if 'avg' in sub_mse_param_sgm_metric:
                        row_list.append(sub_mse_param_sgm_dict[id_n][act][sub_mse_param_sgm_metric])
                
                
                for jsmse_avgsgm_r15_metric in jsmse_avgsgm_r15_dict[id_n][act].keys():
                        row_list.append(jsmse_avgsgm_r15_dict[id_n][act][jsmse_avgsgm_r15_metric])

                writer.writerow(row_list)

# -----------------------fill nan----------------------------
# Define the function to get the real activity name
def get_real_act(full_act):
    if '[' in full_act:  # If there is a bracket in the activity name
        full_act = full_act.split('[')[0]
    return full_act

# Function to convert strings like '[1.0083][1.3766]' into a list of floats
def convert_to_numeric_list(value):
    if isinstance(value, str):
        # Extract all numerical values within square brackets
        numbers = re.findall(r"\d+\.\d+", value)
        return [float(num) for num in numbers] if numbers else np.nan
    return value

# Function to flatten the list of values and take their mean
def flatten_and_average(value):
    if isinstance(value, list):
        if value:
            return np.nanmean(value)  # Take mean if there are valid elements
        else:
            return np.nan
    return value

def count_avg_mean(list_of_strings):
    # 将字符串形式的列表转换为真正的列表
    lists = [ast.literal_eval(item) for item in list_of_strings]

    # 计算每个位置的均值
    mean_list = [round(sum(values) / len(values),4) for values in zip(*lists)]

    mean_list_string = str(mean_list)

    return mean_list_string

def count_avg_mean_nostr(list_of_strings):
    # 将字符串形式的列表转换为真正的列表
    lists = [ast.literal_eval(item) for item in list_of_strings]

    # 计算每个位置的均值
    mean_list = [round(sum(values) / len(values),4) for values in zip(*lists)]

    # mean_list_string = str(mean_list)

    return mean_list


def fill_nan(csv_path, out_path):
    '''
    This function fill 'nan' in the csv,
    and return the avgerage metric of the vaild activities, 
    which is further used in the SVM
    '''
    # Load the CSV file
    # csv_path = './feature_csv/base_dmse_jsmse_hasnan_insect.csv'
    df = pd.read_csv(csv_path)

    # Apply the function to create a new column for the "real" activity
    df['real_activity'] = df['activity'].apply(get_real_act)

    # Get the columns that have numerical data where NaN replacement needs to be done
    numerical_columns = df.columns.difference(['ID', 'activity', 'real_activity'])

    # Replace 'nan' strings with actual np.nan to facilitate calculations
    df.replace('nan', np.nan, inplace=True)

    # Apply the conversion function to all numerical columns
    for col in numerical_columns:
        if col!='mean':
            df[col] = df[col].apply(convert_to_numeric_list)

    # Apply the flatten and average function to all numerical columns
    for col in numerical_columns:
        if col!='mean':
            df[col] = df[col].apply(flatten_and_average)

    # Now, fill NaN values by calculating the mean for each 'real_activity' group
    for col in numerical_columns:
        if col!='mean':
            df[col] = df.groupby('real_activity')[col].transform(lambda x: x.fillna(round(x.mean(),4)))

    #现在解决mean里的nan问题
    #得到不含nan的csv
    filtered_df = df[df['mean'] != '[nan, nan, nan, nan, nan, nan, nan]']

    #针对每个activity取均值
    all_dict = filtered_df.groupby('real_activity')['mean'].apply(list).to_dict()
    # print(all_dict)


    avg_dict = {}
    for key in all_dict.keys():
        avg_dict[key] = count_avg_mean(all_dict[key])

    # Identify rows where 'mean' column has value "[nan, nan, nan, nan, nan, nan, nan]" and replace accordingly
    df['mean'] = df.apply(
        lambda row: avg_dict[row['real_activity']] if row['mean'] == "[nan, nan, nan, nan, nan, nan, nan]" else row['mean'], axis=1
    )

    # 将结果保存为 CSV 文件
    # df.to_csv('./feature_csv/base_dmse_jsmse_final_bird.csv', index=False)
    df.to_csv(out_path, index=False)


    #下面都是为了填充缺失的活动的值的代码-----------------------------------

    # 筛选需要的sub_mse和js_mse特征列
    sub_mse_columns = [col for col in df.columns if "sub_mse" in col]
    js_mse_columns = [col for col in df.columns if "js_mse" in col]

    # 构建字典，key是real_activity，value是list，每个元素为sub_mse和js_mse拼接的列表
    activity_dict = {}
    for _, row in df.iterrows():
        real_activity = row['real_activity']
        sub_mse_list = row[sub_mse_columns].tolist()
        js_mse_list = row[js_mse_columns].tolist()
        mse_list = sub_mse_list + js_mse_list  # 将sub_mse和js_mse拼接成一个list

        if real_activity not in activity_dict:
            activity_dict[real_activity] = []
        activity_dict[real_activity].append(mse_list)


    activity_dict_mean = df.groupby('real_activity')['mean'].apply(list).to_dict()

    final_dict = {}
    for key in activity_dict.keys():
        final_dict[key] = count_avg_mean_nostr(activity_dict_mean[key]) + \
                        [round(sum(values) / len(values),4) for values in zip(*activity_dict[key])]

    # print(final_dict)
    return final_dict


