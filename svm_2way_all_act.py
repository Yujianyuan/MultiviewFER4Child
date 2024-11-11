import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import json
import pandas as pd
import math

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

def get_all_feat(row):
    metric = [] #新加的指标给个权重好一些
    metric = str_to_list(row['mean']) 
    mse_sum = 0
    for scale in range(1,10+1):
        metric = metric + [float(row['sub_mse(avg_sgm,t=1,delta=0,s=%d)'%scale])]
    
    for scale in range(1,5+1):
        metric = metric + [float(row['js_mse(avg_sgm,s=%d,r=0.15)'%scale])]

    return metric


def SVM_2way_all(diag_path, csv_path, all_act, rest_feat):
    diag_dict = json.load(open(diag_path))

    # time = 1 #ASD TD
    # time = 2 #DD TD
    # time = 3 #ASD DD

    for time in [1,2,3]:

        if time == 1:
            remove = 'DD'
            print('A. vs T.')
        elif time == 2:
            remove = 'ASD'
            print('D. vs T.')
        elif time == 3:
            remove = 'TD'
            print('A. vs D.')

        # 读取数据
        df = pd.read_csv(csv_path)

        # 收集每个人的所有活动数据
        person_metrics = {}
        for index, row in df.iterrows():
            frameID = row['ID']
            act = get_real_act(row['activity'])

            # 跳过不要的活动
            if act not in all_act:
                continue

            metric = get_all_feat(row)
            if frameID not in person_metrics:
                person_metrics[frameID] = {}
            if act not in person_metrics[frameID]:
                person_metrics[frameID][act] = []

            person_metrics[frameID][act].append(metric)

        # 筛选出包含所有活动的数据
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

        # 转换数据格式
        data = {
            'ASD': [],
            'DD': [],
            'TD': []
        }

        del data[remove]

        for person, metrics in final_metrics.items():
            for key in diag_dict:
                if key == remove or not '%03d' % person in diag_dict[key]:
                    continue
                data[key].append(metrics)

        # 提取特征和标签
        features = []
        labels = []

        for label, vectors in data.items():
            # label_code = {
            #     "TD": 0,
            #     "ASD": 1 if time in (1, 3) else None,
            #     "DD": 1 if time == 2 else None
            # }[label]

            if time==1:
                if label == "TD":
                    label_code = 0
                elif label == "ASD":
                    label_code = 1
            elif time==2:
                if label == "TD":
                    label_code = 0
                elif label == "DD":
                    label_code = 1
            elif time==3:
                if label == "DD":
                    label_code = 0
                elif label == "ASD":
                    label_code = 1
            
            for vector in vectors:
                features.append(vector)
                labels.append(label_code)

        features = np.array(features)
        labels = np.array(labels)

        # 数据预处理
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        # print(len(features_scaled))

        # 创建 SVM 模型
        svm_model = SVC(kernel='linear', class_weight='balanced', tol=1.5e-2, probability=True)
        loo = LeaveOneOut()

        # 进行交叉验证
        cv_scores = cross_val_score(svm_model, features_scaled, labels, cv=loo)

        # 计算平均准确率
        average_accuracy = np.mean(cv_scores)
        acc = round(100 * average_accuracy, 1)
        print(f'所有活动的平均准确率: {acc}%')


        # # 留一法交叉验证
        # loo = LeaveOneOut()
        # y_true, y_pred, y_probs = [], [], []
        # coef_sum = np.zeros(features_scaled.shape[1])  # 初始化系数和

        # from sklearn.preprocessing import LabelEncoder
        # for train_index, test_index in loo.split(features_scaled):
        #     X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        #     y_train, y_test = labels[train_index], labels[test_index]
        #     #print(y_train)

        #     # if y_train.dtype == object:
        #     #     le = LabelEncoder()
        #     #     y_train = le.fit_transform(y_train)
        #     #     print("Labels transformed by LabelEncoder:", y_train)
            
        #     svm_model.fit(X_train, y_train)
        #     y_probs.append(svm_model.predict_proba(X_test)[:, 1][0])  # 获取正类别的预测概率
        #     y_pred.append(svm_model.predict(X_test)[0])
        #     y_true.append(y_test[0])
        #     #coef_sum += np.abs(svm_model.coef_[0])  # 累加特征权重
        #     # 归一化特征权重并累加
        #     normalized_coefs = np.abs(svm_model.coef_[0]) / np.linalg.norm(svm_model.coef_[0])
        #     coef_sum += normalized_coefs

        # conf_matrix = confusion_matrix(y_true, y_pred)

        # accuracy = accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred)
        # recall = recall_score(y_true, y_pred)
        # sensitivity = recall  # 敏感度和召回率相同
        # specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        # correct_rate = accuracy

        # #print("混淆矩阵:\n", conf_matrix)
        # print('正确率: {:.3f} 精确度: {:.3f} 召回率: {:.3f} 特异度: {:.3f}'.format(correct_rate,precision,recall,specificity))
