import os
import json
import pandas as pd
import numpy as np
import math
import csv
from sigma import *
from dmse import *
from pmse import *
from mfeprob import *
from tools import *
from svm_3way_all_act import *
from svm_2way_all_act import *

if __name__ == '__main__':
    # TODO: the path of the emot file
    info_base = '/data1/shanghai_emot_file/final_emot_clean'
    all_emot = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    all_act = ['Bamboo','RJAduck','IJAduck','peekaboo','RJAposterC','RJAposterD','RJAcar']
    diag_path = './diagnosis_dict.json'

    # compute mProb
    mProb_dict = mFEProb(info_base, all_emot)

    # compute dmse
    avg_dmse_sgm_dict = compute_sigma_mse(info_base, all_act, 'dmse')
    dmse_dict = DMSE(info_base, avg_dmse_sgm_dict, all_emot)

    # compute pmse
    avg_pmse_sgm_dict = compute_sigma_mse(info_base, all_act, 'pmse')
    pmse_dict = PMSE(info_base, avg_pmse_sgm_dict, all_emot)

    # write csv
    hasnan_path = './feature_csv/base_dmse_jsmse_hasnan.csv'
    write_to_csv(mProb_dict, dmse_dict, pmse_dict, hasnan_path)

    # fill nan
    final_path = './feature_csv/base_dmse_jsmse_final.csv'
    valid_avg_metric_dict = fill_nan(hasnan_path, final_path)

    # SVM
    # 3-way
    SVM_3way_all(diag_path, final_path, all_act, valid_avg_metric_dict)

    # 2-way
    SVM_2way_all(diag_path, final_path, all_act, valid_avg_metric_dict)