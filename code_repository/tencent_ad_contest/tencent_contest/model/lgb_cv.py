#! /usr/bin/python3

import numpy as np
import lightgbm as lgb
import pandas as pd

from setting import *
import scipy
from sklearn.model_selection import train_test_split
import gc

cnt = 1
root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data/upsample/'

def main():

    cv_numiterations = params['num_iterations']
    print(params)
    for i in range(cnt):
        train_fea = scipy.sparse.load_npz(root_path + 'train_{}.npz'.format(i))
        train_lab = pd.read_csv(root_path + \
            'label_{}.csv'.format(i)).loc[:, 'label'].values

        # valid_fea = scipy.sparse.load_npz(root_path + 'train_{}.npz'.format((i + 1) % cnt))
        # valid_lab = pd.read_csv(root_path + \
        #     'label_{}.csv'.format((i + 1) % cnt)).loc[:, 'label'].values
        
        lgb_train = lgb.Dataset(train_fea, label=train_lab)
        # lgb_valid = lgb.Dataset(valid_fea, label=valid_lab, reference=lgb_train)

        print('cross-valid cnt={}/{}'.format(i + 1, cnt))

        # solver = lgb.train(cv_params, lgb_train, valid_sets=[lgb_train], \
        #     valid_names=['train'], verbose_eval=True, \
        #     num_boost_round=cv_numiterations, \
        #     early_stopping_rounds=cv_early_stopping_round)
        
        # print(solver.feature_importance())
        lgb.cv(params, lgb_train, \
            verbose_eval=True, \
            num_boost_round=cv_numiterations)




if __name__ == '__main__':
    main()
