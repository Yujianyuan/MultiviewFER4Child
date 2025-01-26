import os
import json
import pandas as pd
import numpy as np
import math
import csv
from scipy.spatial.distance import jensenshannon

def get_min_max(key_list):
    min_num = min(map(int, key_list))
    max_num = max(map(int, key_list))
    return min_num, max_num

def list_sum(list1, list2):
    return [list1[idx] + list2[idx] for idx in range(7)]

def list_abs_sub(list1, list2):
    return [abs(list1[idx] - list2[idx]) for idx in range(7)]

# get sub sequence of time series
def get_sub_seq(emot_dict):
    result_dict = {}
    min_num, max_num = get_min_max(emot_dict.keys())
    
    # Generate a sequence from min_num to max_num-1
    chosen_seq = list(range(min_num, max_num))
    
    for frame in chosen_seq:
        if frame + 1 > max_num:
            continue
        
        # Get data from the current frame and the next frame
        now_list = emot_dict[frame]
        next_list = emot_dict[frame + 1]
        
        # Calculate the absolute difference
        result = list_abs_sub(next_list, now_list)
        
        # Calculate the average change
        result = sum(result) / len(result)
        
        result_dict[frame] = result

    return result_dict

# compute js std of a time series
def compute_js_nanstd(series):
    # compute average distribution
    average_distribution = np.mean(series, axis=0)
    
    # avoid zero devision
    epsilon = 1e-10
    average_distribution += epsilon
    average_distribution /= average_distribution.sum()

    series += epsilon
    series = np.array([p / p.sum() for p in series])

    # compute js distance
    distances = np.array([jensenshannon(p, average_distribution, base=2) for p in series])
    
    squared_distances = distances**2
    
    # compute std
    variance = np.mean(squared_distances)
    out_std = np.sqrt(variance)
    
    return out_std


#compute sigma for mse
def compute_sigma_mse(info_base, all_act, mse_type):
    # info_base = '/data1/shanghai_emot_file/final_emot_clean'
    # all_act=[
    #     'Bamboo','RJAduck','IJAduck',
    #     'peekaboo','RJAposterC','RJAposterD','RJAcar'
    # ]

    feat_dict = {}
    avg_sgm_dict = {}

    id_list = os.listdir(info_base)
    for now_act in all_act:
        avg_sgm_dict[now_act]=[]

        # For each person
        for id_n in id_list:
            # initialization
            if id_n not in feat_dict.keys():
                feat_dict[id_n] = {}

            id_path = os.path.join(info_base,id_n)
            csv_list = os.listdir(id_path)
            for csv_name in csv_list:
                if 'csv' not in csv_name:
                    continue
                
                csv_path = os.path.join(id_path,csv_name)
                activity = csv_name.split('.')[0].split('_')[1][1:-1]

                # only compute current activity
                real_act = get_real_act(activity)
                if real_act != now_act:
                    continue

                if activity not in feat_dict[id_n].keys():
                    feat_dict[id_n][activity] = {}
                else:
                    continue 

                kid_dict = {}
                df = pd.read_csv(csv_path)
                for index, row in df.iterrows():
                    frameID = row['frameID'] # int
                    personID = row['personID']
                    
                    # only compute emotion of the child
                    if personID == 'kid': 
                        emot_list = []
                        for item in all_emot:
                            emot_list.append(row[item])
                        kid_dict[frameID] = emot_list

                # choose child feature
                chosen_dict = kid_dict
                
                emot_sub_seq = get_sub_seq(chosen_dict)

                #avg 
                valid_data = np.array([v for v in emot_sub_seq.values() if not np.all(np.isnan(v))])
                if mse_type == 'dmse':
                    sigma = np.std(valid_data)
                elif mse_type == 'pmse':
                    sigma = compute_js_nanstd(valid_data)

                avg_sgm_dict[now_act].append(sigma)

                print('finish:',id_n,activity)

        #avg
        avg_sgm_dict[now_act] = [sum(avg_sgm_dict[now_act])/len(avg_sgm_dict[now_act])]

    return avg_sgm_dict





















all_emot = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']



def custom_granulate_time_series(time_series, scale):
    """
    根据给定的尺度因子对时间序列进行粗化，并在包含的nan大于50%时返回nan
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

def sample_entropy_nanok(time_series, sample_length, tolerance=None):
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


from pyentrp import entropy as ent

def multiscale_entropy_new_coarse(time_series, max_scale, m, r):
    """
    计算多尺度熵
    :param time_series: 输入的时间序列（一维）
    :param max_scale: 最大尺度因子
    :param m: 嵌入维度
    :param r: 相似度阈值
    :return: 每个尺度的样本熵列表
    """
    mse = []
    for scale in range(1, max_scale + 1):
        # 构造粗化时间序列
        #coarse_grained_ts = ent.util_granulate_time_series(time_series, scale)
        coarse_grained_ts = custom_granulate_time_series(time_series, scale)
        # 计算样本熵, 可以接受输入中有nan
        #se = ent.sample_entropy(coarse_grained_ts, m, r)
        se = sample_entropy_nanok(coarse_grained_ts, m, r)
        mse.append(se[0] if len(se) > 0 else np.nan)  # 处理可能出现的空列表

    return mse

def get_multiscale_entropy_new_coarse(emot_dict, m=2, max_scale=10, r_rate=0.15, mean=False, fix_sgm = -1000):
    # 从字典中提取数据，不去除nan
    valid_data = np.array([v for v in emot_dict.values()]) #if not np.all(np.isnan(v))])
    valid_data_for_std = np.array([v for v in emot_dict.values() if not np.all(np.isnan(v))])

    #sigma = np.std(valid_data)
    #print(valid_data.shape)
    # 计算每个维度的多尺度熵
    #max_scale = 10  # 最大尺度因子
    #m = 2           # 嵌入维度
    #r = r_rate*sigma # 相似度阈值,sigma用序列的标准差

    if fix_sgm!=-1000:#给定的sigma，用于使用群体sigma
        if mean == True:
            sigma = fix_sgm #使用给定的sigma
            r = r_rate*sigma # 相似度阈值,sigma用序列的标准差
            mse_results = multiscale_entropy_new_coarse(valid_data, max_scale, m, r)
            return mse_results
        else:
            return None #还没写

    if mean == True:
        #sigma = np.std(valid_data)
        sigma = np.std(valid_data_for_std)
        r = r_rate*sigma # 相似度阈值,sigma用序列的标准差
        mse_results = multiscale_entropy_new_coarse(valid_data, max_scale, m, r)
    else:
        mse_results = []
        for i in range(7):
            sigma = np.std(valid_data[:, i])
            r = r_rate*sigma # 相似度阈值,sigma用序列的标准差
            mse = multiscale_entropy_new_coarse(valid_data[:, i], max_scale, m, r)
            mse_results.append(mse)


def multiscale_entropy(time_series, max_scale, m, r):
    """
    计算多尺度熵
    :param time_series: 输入的时间序列（一维）
    :param max_scale: 最大尺度因子
    :param m: 嵌入维度
    :param r: 相似度阈值
    :return: 每个尺度的样本熵列表
    """
    mse = []
    for scale in range(1, max_scale + 1):
        # 构造粗化时间序列
        coarse_grained_ts = ent.util_granulate_time_series(time_series, scale)
        # 计算样本熵
        se = ent.sample_entropy(coarse_grained_ts, m, r)
        mse.append(se[0] if len(se) > 0 else np.nan)  # 处理可能出现的空列表

    return mse


def get_multiscale_entropy(emot_dict, m=2, max_scale=10, r_rate=0.15, mean=False, fix_sgm = -1000):
    # 从字典中提取有效数据
    #print(emot_dict)
    valid_data = np.array([v for v in emot_dict.values() if not np.all(np.isnan(v))])

    #sigma = np.std(valid_data)
    #print(valid_data.shape)
    # 计算每个维度的多尺度熵
    #max_scale = 10  # 最大尺度因子
    #m = 2           # 嵌入维度
    #r = r_rate*sigma # 相似度阈值,sigma用序列的标准差

    if fix_sgm!=-1000:#给定的sigma，用于使用群体sigma
        if mean == True:
            sigma = fix_sgm #使用给定的sigma
            r = r_rate*sigma # 相似度阈值,sigma用序列的标准差
            mse_results = multiscale_entropy(valid_data, max_scale, m, r)
            return mse_results
        else:
            return None #还没写

    if mean == True:
        sigma = np.std(valid_data)
        r = r_rate*sigma # 相似度阈值,sigma用序列的标准差
        mse_results = multiscale_entropy(valid_data, max_scale, m, r)
    else:
        mse_results = []
        for i in range(7):
            sigma = np.std(valid_data[:, i])
            r = r_rate*sigma # 相似度阈值,sigma用序列的标准差
            mse = multiscale_entropy(valid_data[:, i], max_scale, m, r)
            mse_results.append(mse)

    return mse_results  # 返回每个维度的多尺度熵结果列表


def get_min_max(key_list):
    min_num = 1000000
    max_num = 0
    for item in key_list:
        if int(item) > max_num:
            max_num = int(item)
        if int(item) < min_num:
            min_num = int(item)
    return min_num, max_num

def list_sum(list1,list2):
    return [list1[idx] + list2[idx] for idx in range(7)]

def list_abs_sub(list1,list2):
    return [abs(list1[idx] - list2[idx]) for idx in range(7)]

#得到后一帧和前一帧的差的绝对值(t=1, delta=0)
#正常情况下是[t-delta,t+delta]
#正常操作，如果做差的其中有一个是nan，输出nan即可。所以不用管它，反正算出来都是nan
def get_multiscale_sub_seq(emot_dict, t=1, delta=0, mean = False):
    result_dict = {}
    min_num, max_num = get_min_max(emot_dict.keys())
    #生成min_num+delta到max_num-delta-t之间的间隔为n的序列
    #这里可能要考虑一下视频太短导致max-delta-t比min+delta小的情况
    chosen_seq = list(range(min_num+delta, max_num-delta-t + 1, t))
    for frame in chosen_seq:
        if frame+t+delta > max_num:
            continue
        #小的部分
        little_sum = 7*[0.0]
        for idx in range(-delta, delta+1):
            now_list = emot_dict[frame+idx]
            little_sum = list_sum(little_sum, now_list)

        #大的部分
        bigger_sum = 7*[0.0]
        for idx in range(-delta, delta+1):
            now_list = emot_dict[frame+t+idx] # +t
            bigger_sum = list_sum(bigger_sum, now_list)

        #做差
        result = list_abs_sub(bigger_sum, little_sum)
        if mean == True: #计算平均变化
            result = sum(result) / len(result)
        result_dict[frame] = result

    return result_dict

#只保留4位小数
def to_four_decimals(input_list):
    return [round(element, 4) for element in input_list]

def get_dict_result_entropy(chosen_dict):
    result_list = [to_four_decimals(mean_prob(chosen_dict)),to_four_decimals(std_prob(chosen_dict)),
                round(mean_entropy(chosen_dict),4)]
    return result_list

def get_dict_result(chosen_dict):
    result_list = [to_four_decimals(mean_prob(chosen_dict)),to_four_decimals(std_prob(chosen_dict))]#,
                #round(mean_entropy(chosen_dict),4)]
    return result_list

def get_real_act(full_act):
    if '[' in full_act:#有的可能后面带一个[a1]
        full_act = full_act.split('[')[0]
    return full_act    



'''
xxx_dict.json

--ID
    --activity
        --feat_name  eg: mse(s=1)

没有明面上的feat_name一样，只有处理后的一样，比如bamboo[a1]和bamboo[a2]
所以这样存储不会存在问题
'''

all_act=[
    'Bamboo','Puzzle1','RJAposterA','RJAposterB','RJAduck','IJAduck',
    'peekaboo','Puzzle2','RJAposterC','RJAposterD','RJAcar','IJAcar','Bubble'
]

'''feat_dict = {}

info_base = '/data1/shanghai_emot_file/final_emot_clean'
id_list = os.listdir(info_base)


avg_sgm_dict = {}
all_sgm_dict = {}

for now_act in all_act:
    #初始化
    avg_sgm_dict[now_act]=[]
    all_sgm_dict[now_act]=[]
    all_smg_total_list = []
    #对每个人
    for id_n in id_list:
        if id_n not in feat_dict.keys():
            feat_dict[id_n] = {}#初始化

        id_path = os.path.join(info_base,id_n)
        csv_list = os.listdir(id_path)
        for csv_name in csv_list:
            if 'csv' not in csv_name:
                continue
            
            csv_path = os.path.join(id_path,csv_name)
            activity = csv_name.split('.')[0].split('_')[1][1:-1]

            real_act = get_real_act(activity)
            if real_act != now_act:#只看当前的activity的
                continue

            if activity not in feat_dict[id_n].keys():
                feat_dict[id_n][activity] = {}#初始化
            else:
                continue #已经有的活动不重复计算了
                print('错误，出现了重复名!!!')

            kid_dict = {}
            df = pd.read_csv(csv_path)
            for index, row in df.iterrows():
                frameID = row['frameID'] #int类型
                personID = row['personID']
                if personID == 'kid': #现在先不考虑老师的情况了
                    emot_list = []
                    for item in all_emot:
                        emot_list.append(row[item])
                    kid_dict[frameID] = emot_list

            #下面准备写特征
            chosen_dict = kid_dict

            # 调参sub mse(all_sigma)用整体的sigma,不过注意不同activity还是要使用不同的sigma
            # case1:每个单独算sigma然后取均值 avg_sgm
            # case2:所有的串起来成一个序列然后计算总均值 all_sgm
            
            sub_seq_mean_dict = get_multiscale_sub_seq(chosen_dict,1,0,mean=True)

            #avg 
            valid_data = np.array([v for v in sub_seq_mean_dict.values() if not np.all(np.isnan(v))])
            sigma = np.std(valid_data)
            avg_sgm_dict[now_act].append(sigma)

            #all 
            all_smg_total_list+=[v for v in sub_seq_mean_dict.values() if not np.all(np.isnan(v))]

            print('finish:',id_n,activity)

    #avg
    avg_sgm_dict[now_act] = [sum(avg_sgm_dict[now_act])/len(avg_sgm_dict[now_act])]

    #all
    print('total_len:',len(all_smg_total_list))
    total_valid_data = np.array(all_smg_total_list)
    total_sigma = np.std(total_valid_data)
    all_sgm_dict[now_act].append(total_sigma)

print('avg_sgm_dict:',avg_sgm_dict)
print('all_sgm_dict:',all_sgm_dict)'''

# avg_sgm_dict: {'Bamboo': [0.037075855151763155], 'Puzzle1': [0.027740756401715288], 'RJAposterA': [0.02971787719008708], 'RJAposterB': [0.033900505525043374], 'RJAduck': [0.029738024298511478], 'IJAduck': [0.031168508039761948], 'peekaboo': [0.04323126105338135], 'Puzzle2': [0.027007937366155314], 'RJAposterC': [0.029149385173110814], 'RJAposterD': [0.030838297074792787], 'RJAcar': [0.030619865010231756], 'IJAcar': [0.0326816186562792], 'Bubble': [0.038613510836280646]}
# all_sgm_dict: {'Bamboo': [0.039111871902462174], 'Puzzle1': [0.02978653130424066], 'RJAposterA': [0.03334125506113402], 'RJAposterB': [0.03886540907259236], 'RJAduck': [0.03573480580317761], 'IJAduck': [0.03352720291080184], 'peekaboo': [0.04548562011195843], 'Puzzle2': [0.028426549555141706], 'RJAposterC': [0.03339139491039148], 'RJAposterD': [0.03533510446466374], 'RJAcar': [0.03385145474517155], 'IJAcar': [0.034344182216425205], 'Bubble': [0.040894438461367]}

avg_sgm_dict={'Bamboo': [0.037075855151763155], 'Puzzle1': [0.027740756401715288], 'RJAposterA': [0.02971787719008708], 'RJAposterB': [0.033900505525043374], 'RJAduck': [0.029738024298511478], 'IJAduck': [0.031168508039761948], 'peekaboo': [0.04323126105338135], 'Puzzle2': [0.027007937366155314], 'RJAposterC': [0.029149385173110814], 'RJAposterD': [0.030838297074792787], 'RJAcar': [0.030619865010231756], 'IJAcar': [0.0326816186562792], 'Bubble': [0.038613510836280646]}
all_sgm_dict={'Bamboo': [0.039111871902462174], 'Puzzle1': [0.02978653130424066], 'RJAposterA': [0.03334125506113402], 'RJAposterB': [0.03886540907259236], 'RJAduck': [0.03573480580317761], 'IJAduck': [0.03352720291080184], 'peekaboo': [0.04548562011195843], 'Puzzle2': [0.028426549555141706], 'RJAposterC': [0.03339139491039148], 'RJAposterD': [0.03533510446466374], 'RJAcar': [0.03385145474517155], 'IJAcar': [0.034344182216425205], 'Bubble': [0.040894438461367]}


feat_dict = {}

info_base = '/data1/shanghai_emot_file/final_emot_clean'
id_list = os.listdir(info_base)
for id_n in id_list:
    if id_n not in feat_dict.keys():
        feat_dict[id_n] = {}#初始化

    id_path = os.path.join(info_base,id_n)
    csv_list = os.listdir(id_path)
    for csv_name in csv_list:
        if 'csv' not in csv_name:
            continue
        
        csv_path = os.path.join(id_path,csv_name)
        activity = csv_name.split('.')[0].split('_')[1][1:-1]

        if activity not in feat_dict[id_n].keys():
            feat_dict[id_n][activity] = {}#初始化
        else:
            continue #已经有的活动不重复计算了
            print('错误，出现了重复名!!!')

        kid_dict = {}
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            frameID = row['frameID'] #int类型
            personID = row['personID']
            if personID == 'kid': #现在先不考虑老师的情况了
                emot_list = []
                for item in all_emot:
                    emot_list.append(row[item])
                kid_dict[frameID] = emot_list

        #下面准备写特征
        chosen_dict = kid_dict

        # 调参sub mse(all_sigma)用整体的sigma,不过注意不同activity还是要使用不同的sigma
        # case1:每个单独算sigma然后取均值 avg_sgm
        # case2:所有的串起来成一个序列然后计算总均值 all_sgm
        real_act = get_real_act(activity)
        for (t,delta) in [(1,0)]:
            sub_seq_mean_dict = get_multiscale_sub_seq(chosen_dict,t,delta,mean=True)
            #avg
            avg_sgm_mse_results=get_multiscale_entropy_new_coarse(sub_seq_mean_dict,
                                mean=True,fix_sgm = avg_sgm_dict[real_act][0])
            scale_num = len(avg_sgm_mse_results)
            for idx in range(scale_num):
                feat_dict[id_n][activity]['sub_mse(avg_sgm,t=%d,delta=%d,s=%d)'%(t,delta,idx+1)]\
                 = to_four_decimals([avg_sgm_mse_results[idx]])
            
            #all
            all_sgm_mse_results=get_multiscale_entropy_new_coarse(sub_seq_mean_dict,
                                mean=True,fix_sgm = all_sgm_dict[real_act][0])
            scale_num = len(all_sgm_mse_results)
            for idx in range(scale_num):
                feat_dict[id_n][activity]['sub_mse(all_sgm,t=%d,delta=%d,s=%d)'%(t,delta,idx+1)] \
                = to_four_decimals([all_sgm_mse_results[idx]])

        print('finish:',id_n,activity)


out_path = './feature_dict/sub_mse_param(sigma,new_coa_sampen)_feat_dict_v1.json'
formatted_json = '{' + custom_format(feat_dict) + '}'
# Save to a file
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(formatted_json)

