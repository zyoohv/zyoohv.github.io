#! /usr/bin/python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shelve
from tqdm import tqdm, trange

static_columns = [
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
    # 'carrier'
    # 'house'    
]

def draw_static(columns_name):

    # loading dataset
    columns_value = df.loc[:, columns_name].values
    fread = shelve.open(hashtable_path)
    dic = fread[columns_name][0]
    fread.close()

    # format: idx, posi, all
    result = np.zeros((len(list(dic.keys())), 3))
    for i in trange(n, desc=columns_name):
        line = str(columns_value[i])
        line = [int(x) for x in line.split(':')]
        l = label_ary[i]
        for val in line:
            if val == 0:
                continue
            result[dic[val] - 1][2] += 1
            if l == 1:
                result[dic[val] - 1][1] += 1
    
    max_x = np.max(result)
    result = result / np.max(max_x)
    output = []
    for key in dic.keys():
        output.append(['{}_'.format(key), result[dic[key] - 1][1], \
            result[dic[key] - 1][2]])
    result = sorted(output, key=lambda x: x[1], reverse=True)
    result = pd.DataFrame(result, columns=['idx', 'posi', 'all'])

    # just show part features...
    y_total = result.shape[0]
    y_show = min(y_maxnum, y_total)
    result = result.head(y_show)

    fig_len = max(10, int(1.0 * result.shape[0] / 50 * 15))
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, fig_len))

    sns.set_color_codes("pastel")
    sns.barplot(x="all", y="idx", data=result, label="Total", color="b")

    sns.set_color_codes("muted")
    sns.barplot(x="posi", y="idx", data=result, label="positive", color="b")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 1), ylabel="{}/{}".format(y_show, y_total), \
        xlabel='{}({})'.format(columns_name, int(max_x)))
    sns.despine(left=True, bottom=True)
    # plt.show()

    plt.savefig('{}.png'.format(columns_name))
    plt.close()


def main():

    for i in range(len(static_columns)):
        draw_static(static_columns[i])


if __name__ == '__main__':
    root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data/'
    train_path = root_path + 'train_assemble.csv'
    hashtable_path = root_path + 'hashtable.txt'
    
    df = pd.read_csv(train_path)
    label_ary = df.loc[:, 'label'].values.astype(int)
    n = df.shape[0]
    y_maxnum = 30
    main()
