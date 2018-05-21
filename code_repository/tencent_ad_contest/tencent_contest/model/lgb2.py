#! /usr/bin/python3

import numpy as np
import lightgbm as lgb
import pandas as pd
from setting import *
import gc

cnt = 19
root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data/'

def main():
    res = []
    num_iterations = params['num_iterations']
    early_stopping_round = params['early_stopping_round']
    print(params)
    for i in range(cnt):
        train_fea = pd.read_csv(root_path + 'train_score_{}.csv'.format(i))
        train_lab = pd.read_csv(root_path + 'label_{}.csv'.format(i))
        train_lab = train_lab.loc[:, 'label'].values

        lgb_train = lgb.Dataset(train_fea, train_lab)

        solver = lgb.train(params, lgb_train, \
                           valid_sets=[lgb_train], \
                           valid_names=['train'], \
                           verbose_eval=True, \
                           num_boost_round=num_iterations, \
                           early_stopping_rounds=early_stopping_round)

        pred_fea = pd.read_csv(root_path + 'res_score.csv')
        pred_fea = pred_fea.drop([i], axis=1).values
        res.append(solver.predict(pred_fea, num_iteration=solver.best_score))
        pd.DataFrame(np.array(res).T).to_csv(root_path + \
                                             'res_score2.csv', index=False)

    res = np.mean(res, axis=0)
    pred_pair = pd.read_csv(root_path + 'test1.csv')
    pred_pair['score'] = res
    pred_pair['score'] = pred_pair['score'].apply(lambda x: '{:.6f}'.format(x))
    pred_pair.to_csv(root_path + 'submission-5000-layer2.csv', index=False)


if __name__ == '__main__':
    main()
