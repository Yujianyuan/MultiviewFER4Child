import numpy as np

#mProb feature
def mFEProb_one(emot_dict):
    '''
    emot_dict -- key: frameID  
                 value: 7 dim facial expression probability of frameID
    '''
    
    sums = np.zeros(7)
    count = 0

    for value in emot_dict.values():
        if not np.all(np.isnan(value)):
            sums += value
            count += 1

    averages = sums / count
    averages = averages.tolist() 
    return averages
