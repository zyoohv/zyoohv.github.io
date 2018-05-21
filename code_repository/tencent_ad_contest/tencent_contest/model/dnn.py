#! /usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function
import pandas as pd
import scipy
import scipy.sparse


root_path = '/home/zyoohv/Documents/tencent_dataset/preliminary_contest_data' \
            '/upsample/'


class datasetutil():
    def __init__(self, index=0):
        '''
        parameter:
        index = 0
        '''

        self.train_fea = scipy.sparse.load_npz(root_path + \
            'train_{}.npz'.format(index)).tocsr()
        self.feature_shape = self.train_fea.shape
        self.label = pd.read_csv(root_path + 'label_{}.csv'.format(index))
        self.label = self.label.loc[:, 'label'].values.astype(int)
        self.pred_fea = scipy.sparse.load_npz(root_path + \
            'pred_{}.npz'.format(index)).tocsr()

        fea_index = list(range(self.fea_shape[0]))
        np.random.shuffle(fea_index)
        split_index = int(0.8 * len(fea_index))
        self.train_index = np.array(fea_index[:split_index])
        self.valid_index = np.array(fea_index[split_index:])

    @property
    def fea_shape(self):
        return self.feature_shape

    def train_sample(self, batch_size):
        select_index = np.random.choice(len(self.train_index), batch_size,
                                        replace=False)
        select_index = self.train_index[select_index]

        train_sample = []
        for idx in select_index:
            fea_line = self.train_fea[idx].toarray()[0]
            fea_line = (fea_line - np.mean(fea_line)) / np.std(fea_line)
            train_sample.append(fea_line)
        train_label = self.label[select_index]

        return train_sample, [int(x) for x in train_label]

    def valid_begin(self):
        self.valid_idx = 0

    def valid_next(self, batch_size):
        if self.valid_idx == len(self.valid_index):
            return None, None
        next_idx = min(len(self.valid_index), self.valid_idx + batch_size)
        select_index = np.arange(self.valid_idx, next_idx)
        select_index = self.valid_index[select_index]
        self.valid_idx = next_idx

        valid_sample = []
        for idx in select_index:
            fea_line = self.train_fea[idx].toarray()[0]
            fea_line = (fea_line - np.mean(fea_line)) / np.mean(fea_line)
            valid_sample.append(fea_line)
        valid_label = self.label[select_index]

        return valid_sample, [int(x) for x in valid_label]

    def pred_begin(self):
        self.pred_idx = 0

    def pred_next(self, batch_size):
        if self.pred_idx == self.pred_fea.shape[0]:
            return None, None
        next_idx = min(self.pred_fea.shape[0], self.pred_idx + batch_size)
        select_index = list(range(self.pred_idx, next_idx))
        self.pred_idx = next_idx

        pred_sample = []
        for idx in select_index:
            fea_line = self.pred_fea[idx].toarray()[0]
            fea_line = (fea_line - np.mean(fea_line)) / np.std(fea_line)
            pred_sample.append(fea_line)

        return pred_sample



class net(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(net, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.dropout(x, 0.8)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, 0.8)
        x = F.log_softmax(self.linear3(x))
        return x


class modelutil():

    def __init__(self, input_size, hidden_size=128, learning_rate=0.01):
        self.cuda = torch.cuda.is_available()

        self.classify_loss = nn.NLLLoss()
        self.model = net(input_size, hidden_size=hidden_size)
        if self.cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=learning_rate, momentum=0.9)

        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else \
            torch.Tensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else \
            torch.LongTensor

    def toVariable(self, tensor_ary, tensor_type):
        tensor = Variable(tensor_type(tensor_ary))
        if self.cuda is True:
            tensor.cuda()
        return tensor

    def train_model(self, input_fea, input_label):
        self.model.train(True)
        input_fea = self.toVariable(input_fea, self.FloatTensor)
        # input_label = [x[0] for x in input_label]
        # print('input_label=', input_label)
        # print('type=', type(input_label), type(input_label[0]))
        input_label = self.toVariable(input_label, self.LongTensor)

        self.optimizer.zero_grad()
        train_res = self.model(input_fea)

        loss = self.classify_loss(train_res, input_label)
        loss.backward()
        self.optimizer.step()

        return np.mean(loss.data.cpu().numpy())

    def valid_model(self, input_fea, input_label):
        self.model.train(False)
        input_fea = self.toVariable(input_fea, self.FloatTensor)
        input_label = self.toVariable(input_label, self.LongTensor)

        valid_res = self.model(input_fea)
        loss = self.classify_loss(valid_res, input_label)

        return np.sum(loss.data.cpu().numpy())

    def predict(self, input_fea):
        self.model.train(False)
        input_fea = self.toVariable(input_fea, self.FloatTensor)
        pred_res = self.model(input_fea)

        res_log = pred_res.data.cpu().squeeze().numpy()
        res = []
        for pair in res_log:
            if pair[0] > pair[1]:
                res.append(1. - np.exp(pair[0]))
            else:
                res.append(np.exp(pair[1]))

        return res


def main():

    cnt = 1
    epoch = 200
    valid = 50
    batch_size = 128
    res = []

    for idx in range(cnt):
        datasetutil_now = datasetutil(idx)
        n = datasetutil_now.fea_shape[1]
        model_now = modelutil(input_size=n, hidden_size=128,
                              learning_rate=0.01)

        for i in range(1, epoch + 1):
            train_fea, train_label = datasetutil_now.train_sample(batch_size)
            loss = model_now.train_model(train_fea, train_label)

            print('epoch={}, loss={}'.format(i, loss))

            if i % valid == 0:
                loss_sum = 0.
                loss_cnt = 0
                datasetutil_now.valid_begin()
                while True:
                    train_fea, train_label = datasetutil_now.valid_next(batch_size)
                    if train_fea is None:
                        break
                    loss_cnt += len(train_fea)
                    loss_sum += model_now.train_model(train_fea, train_label)
                print('valid loss = {}'.format(loss_sum / loss_cnt))

        # predict result
        res_tmp = []
        datasetutil_now.pred_begin()
        while True:
            train_fea = datasetutil_now.pred_next(batch_size)
            if train_fea is None:
                break
            res_tmp += model_now.predict(train_fea)
        res.append(res_tmp)

    res = np.mean(res, axis=0)

    pred_pair = pd.read_csv(root_path + '../test1.csv')
    pred_pair['score'] = res
    pred_pair['score'] = pred_pair['score'].apply(lambda x: '{:.6f}'.format(x))
    pred_pair.to_csv(root_path + 'submission-nn.csv', index=False)


if __name__ == '__main__':
    main()