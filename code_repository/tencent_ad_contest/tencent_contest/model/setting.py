#! /usr/bin/python3

param_list = [
{
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'train_metric': True,
    'num_leaves': 63,
    'lambda_l1': 0,
    'lambda_l2': 1,
    # 'min_data_in_leaf': 100,
    'min_child_weight': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'verbose': 1,
    # 'early_stopping_round': 50,
    'num_iterations': 300,
    # 'is_unbalance': True,
    'num_threads': 30
},
]

params = \
{
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'train_metric': True,
    'num_leaves': 63,
    'lambda_l1': 0,
    'lambda_l2': 1,
    # 'min_data_in_leaf': 100,
    'min_child_weight': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'verbose': 1,
    'early_stopping_round': 1000,
    'num_iterations': 5000,
    # 'is_unbalance': True,
    'num_threads': -1
}

