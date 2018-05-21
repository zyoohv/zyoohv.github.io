#! /usr/bin/python3


import xlearn as xl

root_path = '../../tencent_dataset/preliminary_contest_data/'

ffm_model = xl.create_ffm()
ffm_model.setTrain(root_path + 'train.ffm')
ffm_model.setValidate(root_path + 'valid.ffm')


param = {
    'task': 'binary',
    'lr': 0.1,
    'lambda': 0.002,
    'epoch': 20,
    'metric': 'auc'
}

ffm_model.fit(param, root_path + 'ffm_model.out')


ffm_model.setTest(root_path + 'pred.ffm')
ffm_model.predict(root_path + 'ffm_model.out', root_path + 'output.txt')