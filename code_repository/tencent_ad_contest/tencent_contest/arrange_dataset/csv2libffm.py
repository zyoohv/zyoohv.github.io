#! /usr/bin/python3

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scipy
from tqdm import tqdm
import argparse

root_path = '../../tencent_dataset/preliminary_contest_data/'


def trainpred_pair(index):
    train_ary = scipy.sparse.load_npz(root_path + 'train_{}.npz'.format(index))
    train_ary = train_ary.tocsr()
    label_ary = pd.read_csv(root_path + 'label_{}.csv'.format(index))
    label_ary = label_ary.loc[:, 'label'].values

    train_ffm = []
    with tqdm(total=train_ary.shape[0]) as pbar:
        for l, line in zip(label_ary, train_ary):
            now = str(l)
            line = line.toarray()[0]
            for i, val in enumerate(line):
                if float(val) != 0.:
                    now = now + ' {}:{}:{}'.format(1, i + 1, val)
            train_ffm.append(now)
            pbar.update(1)
    
    with open(root_path + 'ffm/train_{}.ffm'.format(index), 'w') as fout:
        for line in train_ffm:
            fout.write(line + '\n')

    pred_ary = scipy.sparse.load_npz(root_path + 'pred_{}.npz'.format(index))
    pred_ary = pred_ary.tocsr()
    pred_ffm = []
    with tqdm(total=pred_ary.shape[0]) as pbar:
        for line in pred_ary:
            line = line.toarray()[0]
            now = ''
            for i, val in enumerate(line):
                if float(val) != 0.:
                    now = now + '{}:{}:{} '.format(1, i + 1, val)
            pred_ffm.append(now)
            pbar.update(1)
    
    with open(root_path + 'ffm/pred_{}.ffm'.format(index), 'w') as fout:
        for line in pred_ffm:
            fout.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, help='produce the index of files')
    args = parser.parse_args()

    trainpred_pair(args.index)


if __name__ == '__main__':
    main()