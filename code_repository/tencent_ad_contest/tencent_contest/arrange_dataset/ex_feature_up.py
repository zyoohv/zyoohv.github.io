#! /usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import gc
import shelve
import scipy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, Lock, Process

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
import sys
sys.stdout = Unbuffered(sys.stdout)

root_path = '../../tencent_dataset/preliminary_contest_data/'

# origin file
train_path = root_path + 'train.csv'
adFeature_path = root_path + 'adFeature.csv'
test_path = root_path + 'test1.csv'
userFeature_path = root_path + 'userFeature.csv'

# tool file
assemble_path = root_path + 'assemble.csv'

# output prefix
# train_out_path = root_path + 'train'
# pred_out_path = root_path + 'pred_full.csv'


def load_assemble():
    if os.path.exists(assemble_path):
        assemble = pd.read_csv(assemble_path)
    else:
        ad_feature = pd.read_csv(adFeature_path)
        
        train = pd.read_csv(train_path)
        train.loc[train['label'] == -1, 'label'] = 0
        
        test = pd.read_csv(test_path)
        test['label'] = -1

        user_feature = pd.read_csv(userFeature_path)

        assemble = pd.concat([train, test])
        assemble = pd.merge(assemble, ad_feature, on='aid', how='left')
        assemble = pd.merge(assemble, user_feature, on='uid', how='left')
        assemble = assemble.fillna('0')
        assemble.to_csv(assemble_path, index=False)
    return assemble

def load_hashtable(hashtable_path, assemble):
    if os.path.exists(hashtable_path):
        os.system('rm {}'.format(hashtable_path))
    
    label_ary = assemble.loc[:, 'label'].values.astype(int)
    assemble = assemble.drop(['uid', 'label'], axis=1)

    dbase = shelve.open(hashtable_path)
    for columns_name in assemble.columns.values:
        columns_val = assemble.loc[:, columns_name].values
        table_tmp = {}
        table_idx = 1
        for l, line in zip(label_ary, columns_val):
            val_list = [int(x) for x in str(line).split(' ')]
            for val in val_list:
                if val == 0:
                    continue
                if val not in table_tmp:
                    table_tmp[val] = [table_idx, 0., 0.]
                    table_idx += 1
                if l == -1:
                    continue
                idx = 1 if l == 1 else 2
                table_tmp[val][idx] += 1
        
        dbase[columns_name] = table_tmp
    dic = dict(dbase)
    dbase.close()
    return dic


###############################################################################
# union feature method.
unioncolumns = [
    ('aid', 'age'),
    ('aid', 'gender'),
    ('aid', 'education'),
    ('aid', 'consumptionAbility'),
    ('aid', 'carrier'),
    ('aid', 'house')
]

def convert_union(assemble, columns_pair):

    dic1 = {}
    dic2 = {}
    columns_fea = assemble.loc[:, list(columns_pair) + ['label']].values

    for line in columns_fea:

        if line[2] == -1:
            continue

        if line[1] not in dic1:
            dic1[line[1]] = [0., 0.]
        idx = 0 if line[2] == 1 else 1
        dic1[line[1]][idx] += 1

        if line[0] not in dic2:
            dic2[line[0]] = [0., 0.]
        idx = 0 if line[2] == 1 else 1
        dic2[line[0]][idx] += 1

    trans_fea = []
    for line in columns_fea:
        ratio1 = dic2[line[0]][0] / np.sum(dic2[line[0]])
        ratio2 = dic1[line[1]][0] / np.sum(dic1[line[1]])
        union_posi = dic2[line[0]][0] + dic1[line[1]][0]
        union_nega = np.sum(dic2[line[0]]) + np.sum(dic1[line[1]])
        union_ratio = union_posi / union_nega

        trans_fea.append([ratio1, ratio2, union_posi, union_nega, union_ratio])
    
    columns_list = ['{}-{}-{}'.format(columns_pair[0], columns_pair[1], i) for \
        i in range(len(trans_fea[0]))]
    print('convert union column \'{}\' finished!'.format(str(columns_pair)))
    return pd.DataFrame(trans_fea, columns=columns_list)



###############################################################################
#embedding method...
embedding = [
    # 'advertiserId',
    # 'campaignId',
    # 'creativeId',
    # 'creativeSize',
    # 'adCategoryId',
    # 'productId',
    # 'productType',
    # 'age',
    # 'gender',
    'marriageStatus',
    # 'education',
    # 'consumptionAbility',
    # 'LBS',
    # 'interest1',
    # 'interest2',
    # 'interest3',
    # 'interest4',
    # 'interest5',
    # 'kw1',
    # 'kw2',
    # 'kw3',
    # 'topic1',
    # 'topic2',
    # 'topic3',
    # 'appIdInstall',
    # 'appIdAction'
    'ct',
    'os',
    # 'carrier',
    # 'house'
]

def convert_embedding(assemble, hashtable, columns_name):

    dic = hashtable[columns_name]
    keys_num = len(list(dic.keys()))

    columns_fea = assemble.loc[:, columns_name].values
    trans_fea = []
    for line in columns_fea:
        trans_fea.append([0 for _ in range(keys_num)])
        val_list = [int(x) for x in str(line).split(' ')]
        for val in val_list:
            if val == 0 or val not in dic:
                continue
            trans_fea[-1][dic[val][0] - 1] = 1
    
    columns_list = ['{}-emb{}'.format(columns_name, i) for i in range(keys_num)]

    print('convert embedding column \'{}\' finished!'.format(columns_name))
    return pd.DataFrame(trans_fea, columns=columns_list)
                

###############################################################################
# count the number of categories...
countnumcolumns = [
    # 'advertiserId',
    # 'campaignId',
    # 'creativeId',
    # 'creativeSize',
    # 'adCategoryId',
    # 'productId',
    # 'productType',
    # 'age',
    # 'gender',
    'marriageStatus',
    # 'education',
    # 'consumptionAbility',
    # 'LBS',
    'interest1',
    'interest2',
    'interest3',
    'interest4',
    'interest5',
    'kw1',
    'kw2',
    'kw3',
    'topic1',
    'topic2',
    'topic3',
    'appIdInstall',
    'appIdAction',
    'ct',
    'os',
    # 'carrier',
    # 'house'
]

def convert_countnum(assemble, columns_name):

    columns_fea = assemble.loc[:, columns_name].values
    trans_fea = []
    for line in columns_fea:
        line = str(line)
        if line == '0':
            trans_fea.append([0])
        else:
            trans_fea.append([len(line.split(' '))])

    print('convert countnum column \'{}\' finished!'.format(columns_name))
    return pd.DataFrame(trans_fea, columns=['{}-ct'.format(columns_name)])
    

###############################################################################
# dictionary learning...
dictionary_columns = {
    # 'advertiserId',
    # 'campaignId',
    # 'creativeId',
    # 'creativeSize',
    # 'adCategoryId',
    # 'productId',
    # 'productType',
    # 'age',
    # 'gender',
    # 'marriageStatus',
    # 'education',
    # 'consumptionAbility',
    # 'LBS',
    'interest1': (20, 5),
    'interest2': (10, 5),
    'interest3': (10, 5),
    'interest4': (10, 5),
    'interest5': (20, 5),
    'kw1': (50, 10),
    'kw2': (50, 10),
    'kw3': (50, 10),
    'topic1': (30, 15),
    'topic2': (30, 15),
    'topic3': (30, 15),
    'appIdInstall': (40, 10),
    'appIdAction': (20, 5),
    # 'ct': 10,
    # 'os': 10,
    # 'carrier': 10,
    # 'house'
}

def convert_dictionary(assemble, hashtable, columns_name, hold_fea_num, trans_fea_num):
    '''
    using dictionary learning to reduce the dimension.
    '''
    dic = hashtable[columns_name]

    hold_fea = [dic[key][:2] for key in dic.keys()]
    hold_fea = sorted(hold_fea, key=lambda x: x[1], reverse=True)
    hold_fea = [x[0] for x in hold_fea[:hold_fea_num]]
    new_table = {}
    new_index = 1
    for val in hold_fea:
        new_table[val] = new_index
        new_index += 1

    columns_fea = assemble.loc[:, columns_name].values
    origin_fea = []
    for line in columns_fea:
        val_list = [int(x) for x in str(line).split(' ')]
        origin_fea.append([0 for _ in range(hold_fea_num)])
        for val in val_list:
            if val == 0 or val not in dic or dic[val] not in hold_fea:
                continue
            origin_fea[-1][new_table[dic[val]] - 1] = 1.

    origin_columns = ['{}-orig{}'.format(columns_name, i) for \
        i in range(hold_fea_num)]
    origin_df = pd.DataFrame(origin_fea, columns=origin_columns)

    # alg = MiniBatchDictionaryLearning(n_components=\
    #     trans_fea_num, alpha=1, n_iter=50, n_jobs=20, \
    #     batch_size=32, verbose=True)
    # print('training...')
    # diso = alg.fit(origin_fea[:1000])
    # print('predicting...')
    # trans_fea = diso.transform(origin_fea)
    # print('finished!')
    # columns_list = ['{}-dl({})'.format(columns_name, i) for \
    #     i in range(trans_fea_num)]
    # trans_df = pd.DataFrame(trans_fea, columns=columns_list)

    print('convert dictionary column \'{}\' finished!'.format(columns_name))
    # return pd.concat([origin_df, trans_df], axis=1)
    return origin_df


###############################################################################
# process interest1-5, kw1-3, topic1-3
ratiocolumns = [
    'aid',
    # 'advertiserId',
    # 'campaignId',
    # 'creativeId',
    # 'creativeSize',
    # 'adCategoryId',
    # 'productId',
    # 'productType',
    # 'age',
    # 'gender',
    'marriageStatus',
    # 'education',
    # 'consumptionAbility',
    # 'LBS',
    'interest1',
    'interest2',
    'interest3',
    'interest4',
    'interest5',
    'kw1',
    'kw2',
    'kw3',
    'topic1',
    'topic2',
    'topic3',
    'appIdInstall',
    'appIdAction',
    'ct',
    'os',
    # 'carrier',
    # 'house'
]

def process_ratio(assemble, hashtable, columns_name):

    dic = hashtable[columns_name]

    columns_fea = assemble.loc[:, columns_name].values
    fea_tmp = []
    for line in columns_fea:
        line = [int(x) for x in str(line).split(' ')]
        posi_num = 0.
        nega_num = 0.
        ratio_sum = 0.
        sum_ratio = 0.
        for val in line:
            if val == 0 or val not in dic:
                continue
            posi_num += dic[val][1]
            nega_num += dic[val][2]
            ratio_sum += posi_num / (nega_num + 1)
        sum_ratio = posi_num / (posi_num + nega_num + 1.)
        fea_tmp.append([posi_num, nega_num, ratio_sum, sum_ratio])
    
    columns_list = ['{}-{}'.format(columns_name, i) for \
        i in range(len(fea_tmp[0]))]
    
    print('process ratio column \'{}\' finished!'.format(columns_name))
    fea_tmp = pd.DataFrame(fea_tmp, columns=columns_list)
    return fea_tmp

###############################################################################
# ConvertVector method
cvcolumns = [
    # 'advertiserId',
    # 'campaignId',
    # 'creativeId',
    # 'creativeSize',
    # 'adCategoryId',
    # 'productId',
    # 'productType',
    # 'age',
    # 'gender',
    'marriageStatus',
    # 'education',
    # 'consumptionAbility',
    # 'LBS',
    'interest1',
    'interest2',
    'interest3',
    'interest4',
    'interest5',
    'kw1',
    'kw2',
    'kw3',
    'topic1',
    'topic2',
    'topic3',
    'appIdInstall',
    'appIdAction',
    # 'ct',
    # 'os',
    # 'carrier',
    # 'house'
]

def convert_vector(assemble, train_df, pred_df, columns_name):

    columns_fea = assemble.loc[:, columns_name].values.astype(np.str)
    cv = CountVectorizer()
    cv.fit(columns_fea)

    train_fea = cv.transform(train_df.loc[:, columns_name].values.astype(np.str))
    pred_fea = cv.transform(pred_df.loc[:, columns_name].values.astype(np.str))

    n = train_fea.shape[1]
    print('convert vector column \'{}.{}\' finished!'.format(columns_name, n))
    return train_fea, pred_fea


###############################################################################
# tranlate to onehot format...
trans2onehot = [
    'aid',
    'advertiserId',
    'campaignId',
    'creativeId',
    'creativeSize',
    'adCategoryId',
    'productId',
    'productType',
    'age',
    'gender',
    # 'marriageStatus',
    'education',
    'consumptionAbility',
    'LBS',
    # 'interest1',
    # 'interest2',
    # 'interest3',
    # 'interest4',
    # 'interest5',
    # 'kw1'
    # 'kw2'
    # 'kw3'
    # 'topic1'
    # 'topic2'
    # 'topic3'
    # 'appIdInstall'
    # 'appIdAction'
    # 'ct',
    # 'os',
    'carrier',
    'house'
]

def convert_onehot(assemble, train_df, pred_df, columns_name):

    columns_fea = assemble.loc[:, [columns_name]].values
    enc = OneHotEncoder()
    enc.fit(columns_fea)

    train_fea = train_df.loc[:, [columns_name]].values
    train_fea = enc.transform(train_fea)

    pred_fea = pred_df.loc[:, [columns_name]].values
    pred_fea = enc.transform(pred_fea)

    print('convert onehot column \'{}\' finished!'.format(columns_name))
    return train_fea, pred_fea


def extract_feature(assemble_now, hashtable, index):

    fea_output = pd.DataFrame()

    for columns_pair in unioncolumns:
        fea_output = pd.concat([fea_output, \
            convert_union(assemble_now, columns_pair)], axis=1)

    for columns_name in embedding:
        fea_output = pd.concat([fea_output, \
            convert_embedding(assemble_now, hashtable, columns_name)], axis=1)

    for columns_name in countnumcolumns:
        fea_output = pd.concat([fea_output, \
            convert_countnum(assemble_now, columns_name)], axis=1)

    for columns_name in dictionary_columns.keys():
        fea_output = pd.concat([fea_output, \
            convert_dictionary(assemble_now, hashtable, columns_name, \
            dictionary_columns[columns_name][0], \
            dictionary_columns[columns_name][1])], axis=1)
    
    for columns_name in ratiocolumns:
        fea_output = pd.concat([fea_output, \
            process_ratio(assemble_now, hashtable, columns_name)], axis=1)
    
    # split fea_output to train_fea and pred_fea.
    fea_output = pd.concat([fea_output, assemble_now.loc[:, ['label']]], axis=1)
    train_fea = fea_output.loc[fea_output['label'] != -1, :].reset_index(drop=True)
    train_fea = train_fea.drop(['label'], axis=1).values
    pred_fea = fea_output.loc[fea_output['label'] == -1, :].reset_index(drop=True)
    pred_fea = pred_fea.drop(['label'], axis=1).values

    # make train_df and pred_df.
    train_df = assemble_now.loc[assemble_now['label'] != -1, :].reset_index(drop=True)
    pred_df = assemble_now.loc[assemble_now['label'] == -1, :].reset_index(drop=True)
    
    # make label array.
    label_ary = train_df.loc[:, ['label']]

    for columns_name in cvcolumns:
        train_spr, pred_spr = convert_vector(assemble_now, \
            train_df, pred_df, columns_name)
        if train_fea.shape[0] == 0:
            train_fea = train_spr
            pred_fea = pred_spr
        else:
            train_fea = scipy.sparse.hstack([train_fea, train_spr])
            pred_fea = scipy.sparse.hstack([pred_fea, pred_spr])
    
    for columns_name in trans2onehot:
        train_spr, pred_spr = convert_onehot(assemble_now, \
            train_df, pred_df, columns_name)
        train_fea = scipy.sparse.hstack([train_fea, train_spr])
        pred_fea = scipy.sparse.hstack([pred_fea, pred_spr])

    print('[{}]train dataset shape='.format(index), train_fea.shape)
    print('[{}]pred dataset shape='.format(index), pred_fea.shape)

    scipy.sparse.save_npz(root_path + 'upsample/train_{}.npz'.format(index), train_fea)
    scipy.sparse.save_npz(root_path + 'upsample/pred_{}.npz'.format(index), pred_fea)
    label_ary.to_csv(root_path + 'upsample/label_{}.csv'.format(index), index=False)


def main():

    assemble = load_assemble()
    posi_df = assemble.loc[assemble['label'] == 1, :].reset_index(drop=True)
    nega_df = assemble.loc[assemble['label'] == 0, :].reset_index(drop=True)
    cnt = int(nega_df.shape[0] / posi_df.shape[0])

    pred_df = assemble.loc[assemble['label'] == -1, :].reset_index(drop=True)
    step_len = posi_df.shape[0]

    for i in range(cnt):
        print('{}/{}'.format(i + 1, cnt))

        start = step_len * i
        end = min((i + 1) * step_len, nega_df.shape[0])
        assemble_now = pd.concat([posi_df, nega_df[start:end], pred_df]).\
            reset_index(drop=True)
        hashtable = load_hashtable(root_path + \
            'upsample/hashtable_{}.txt'.format(i), assemble_now)
        
        extract_feature(assemble_now, hashtable, i)
        
        gc.collect()


if __name__ == '__main__':
    main()