import numpy as np

#mProb feature
def mFEProb_one(emot_dict):
    sums = np.zeros(7)
    count = 0

    for value in emot_dict.values():
        if not np.all(np.isnan(value)):
            sums += value
            count += 1

    averages = sums / count
    averages = averages.tolist() 
    return averages

def to_four_decimals(input_list):
    return [round(element, 4) for element in input_list]


def mFEProb(info_base, all_emot):
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

            feat_dict[id_n][activity]['mean'] = to_four_decimals(mFEProb_one(chosen_dict))

            print('finish:', id_n, activity)

    return feat_dict
