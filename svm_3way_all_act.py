import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score


def str_to_list(str_list):
    """
    Convert a string representation of a list to an actual list of integers.

    :param str_list: A string representing a list, e.g., "[1, 2, 3]"
    :return: A list of integers.
    """
    # Strip the string of brackets and split by comma
    cleaned_str = str_list.strip('[]')
    # Split the string by commas and convert each substring to an integer
    list_of_ints = [float(item.strip()) for item in cleaned_str.split(',') if item.strip()]
    
    return list_of_ints

def get_real_act(full_act):
    if '[' in full_act:#有的可能后面带一个[a1]
        full_act = full_act.split('[')[0]
    return full_act

def normalize_list(lst):
    total = sum(lst)
    if total == 0:
        return [0 for _ in lst]  # Prevent division by zero
    return [x / total for x in lst]


def get_all_feat(row):
    metric = [] 
    metric = str_to_list(row['mean']) 
    mse_sum = 0
    for scale in range(1,10+1):
        metric = metric + [float(row['sub_mse(avg_sgm,t=1,delta=0,s=%d)'%scale])]
    
    for scale in range(1,5+1):
        metric = metric + [float(row['js_mse(avg_sgm,s=%d,r=0.15)'%scale])]

    return metric
    

def SVM_3way_all(diag_path, csv_path, all_act, rest_feat)
    diag_dict = json.load(open(diag_path))

    #注意，有的人可能缺少某些act
    # 读取数据
    df = pd.read_csv(csv_path)

    # 初始化存储每个人所有活动数据的字典
    person_metrics = {}

    for index, row in df.iterrows():
        frameID = row['ID']
        act = get_real_act(row['activity'])

        # 检查当前行活动是否在需要的列表中
        if act not in all_act:
            continue

        metric = get_all_feat(row)
        if frameID not in person_metrics:
            person_metrics[frameID] = {}
        if act not in person_metrics[frameID]:
            person_metrics[frameID][act] = []

        person_metrics[frameID][act].append(metric)


    # 筛选并补充具有所有活动数据的人
    final_metrics = {}
    for person, activities in person_metrics.items():
        # 检查并补充缺失的活动
        for act in all_act:
            if act not in activities:
                # 使用 rest_feat 中的默认值来补充缺失的活动数据
                activities[act] = [np.array(rest_feat[act])]

        # 重新计算平均值并组合数据
        combined_metrics = [np.mean(activities[act], axis=0) for act in all_act]
        final_metrics[person] = np.concatenate(combined_metrics)

    # 准备分类数据
    data = {
        'ASD': [],
        'DD': [],
        'TD': []
    }

    for person, metrics in final_metrics.items():
        for key in diag_dict:
            if '%03d' % person in diag_dict[key]:
                data[key].append(metrics)

    # 提取特征和标签
    features = []
    labels = []

    for label, vectors in data.items():
        label_code = {"ASD": 0, "DD": 1, "TD": 2}[label]
        features.extend(vectors)
        labels.extend([label_code] * len(vectors))

    features = np.array(features)
    labels = np.array(labels)

    # 数据预处理
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(len(features))

    # 创建 SVM 模型
    svm_model = SVC(kernel='linear', class_weight='balanced', tol=1.5e-2, probability=True)

    # 进行交叉验证
    loo = LeaveOneOut()
    cv_scores = cross_val_score(svm_model, features_scaled, labels, cv=loo)

    # 计算平均准确率
    average_accuracy = np.mean(cv_scores)
    acc = round(100 * average_accuracy, 1)
    print(f'整体平均准确率: {acc}%')


