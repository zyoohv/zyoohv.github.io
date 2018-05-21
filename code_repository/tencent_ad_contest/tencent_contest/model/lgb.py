#! /usr/bin/python3

import numpy as np
import lightgbm as lgb
import pandas as pd

from setting import *
import scipy
import scipy.sparse
import gc
import os

cnt = 19
root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data/'

def main():
    num_iterations = params['num_iterations']
    early_stopping_round = params['early_stopping_round']
    print(params)
    for i in range(cnt):
        train_fea = scipy.sparse.load_npz(root_path + 'train_{}.npz'.format(i))
        train_lab = pd.read_csv(root_path + \
            'label_{}.csv'.format(i)).loc[:, 'label'].values

        valid_fea = scipy.sparse.load_npz(root_path + 'train_{}.npz'.format((i + 1) % cnt))
        valid_lab = pd.read_csv(root_path + \
            'label_{}.csv'.format((i + 1) % cnt)).loc[:, 'label'].values

        lgb_train = lgb.Dataset(train_fea, label=train_lab)
        lgb_valid = lgb.Dataset(valid_fea, label=valid_lab, reference=lgb_train)

        print('training cnt={}/{}'.format(i + 1, cnt))

        solver = lgb.train(params, lgb_train, \
            valid_sets=[lgb_train, lgb_valid], \
            valid_names=['train', 'valid'], \
            verbose_eval=True, \
            num_boost_round=num_iterations, \
            early_stopping_rounds=early_stopping_round)

        pred_fea = scipy.sparse.load_npz(root_path + 'pred_{}.npz'.format(i))
        pred_label = solver.predict(pred_fea, num_iteration=solver.best_iteration)
        if os.path.exists(root_path + 'res_score.csv'):
            res = list(pd.read_csv(root_path + 'res_score.csv').values.T)
        else:
            res = []
        res.append(pred_label)
        pd.DataFrame(np.array(res).T).to_csv(root_path + \
            'res_score.csv', index=False)

        for j in range(cnt):
            if j == i:
                continue
            pred_fea = scipy.sparse.load_npz(root_path + 'train_{}.npz'.format(j))
            pred_label = solver.predict(pred_fea, num_iteration=solver.best_iteration)
            if os.path.exists(root_path + 'train_score_{}.csv'.format(j)):
                train_res = list(pd.read_csv(root_path + \
                                'train_score_{}.csv'.format(j)).values.T)
            else:
                train_res = []
            train_res.append(pred_label)
            pd.DataFrame(np.array(train_res).T).to_csv(root_path + \
                'train_score_{}.csv'.format(j), index=False)
        gc.collect()

    res = np.mean(res, axis=0)
    pred_pair = pd.read_csv(root_path + 'test1.csv')
    pred_pair['score'] = res
    pred_pair['score'] = pred_pair['score'].apply(lambda x: '{:.6f}'.format(x))
    pred_pair.to_csv(root_path + 'submission-5000.csv', index=False)



if __name__ == '__main__':
    main()
