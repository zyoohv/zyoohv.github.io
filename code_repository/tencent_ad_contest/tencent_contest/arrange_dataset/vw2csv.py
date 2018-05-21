#! /usr/bin/python3

from tqdm import tqdm
import numpy as np
import pandas as pd
import os

root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data/'
input_path = root_path +  'userFeature.data'
output_path = root_path + 'userFeature.csv'

os.system('rm {}'.format(output_path))

output_file = []
default_value = 0
sep_character = ' '
columns_list = ['uid', 'age', 'gender', 'marriageStatus', 'education', \
                'consumptionAbility', 'LBS', 'interest1', 'interest2', \
                'interest3', 'interest4', 'interest5', 'kw1', 'kw2', \
                'kw3', 'topic1', 'topic2', 'topic3', 'appIdInstall', \
                'appIdAction', 'ct', 'os', 'carrier', 'house']

def main():
    
    user_fea = []
    fea_len = len(columns_list)

    with open(input_path, 'r') as fin:
        for i, line in enumerate(fin):
            default_fea = [0 for _ in range(fea_len)]
            line = line.strip().split('|')
            columns_index = 0
            for item in line:
                item = item.split(' ')
                while columns_list[columns_index] != item[0]:
                    columns_index += 1
                num_ary = sorted([int(x) for x in item[1:]])
                default_fea[columns_index] = ' '.join([str(x) for x in num_ary])
            user_fea.append(default_fea)
            if i % 1000000 == 0:
                print(i, end=',')
    print('')

    userFeature_df = pd.DataFrame(user_fea, columns=columns_list)
    userFeature_df.to_csv(output_path, index=False)
        

if __name__ == '__main__':
    main()