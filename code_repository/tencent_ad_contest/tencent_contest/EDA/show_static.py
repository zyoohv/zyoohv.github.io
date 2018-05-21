#! /usr/bin/python3

import shelve

def main():
    dic = shelve.open(hashtable_path)
    key_list = list(dic.keys())

    for key in key_list:
        hash_num = len( list(dic[key].keys()) )
        print('{}:{}'.format(key, hash_num))


if __name__ == '__main__':
    root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data/'
    hashtable_path = root_path + 'hashtable.txt'

    main()