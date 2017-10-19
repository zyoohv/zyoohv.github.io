import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
from torchvision import models, transforms
import matplotlib.pyplot as plt
from model import model


class addtrain():

    def __init__(self, config):
        self.config = config
        self.model = model(config)
        self.criterion = nn.PairwiseDistance()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])

        self.generate_dataset()


    def generate_dataset(self):
        self.dataset = []
        for _ in range(self.config['generate_dataset']):
            x = (np.random.rand(self.config['n']) - 0.5) * self.config['L']
            y = [np.sum(x)]
            self.dataset.append([x, y])
        self.dataset = np.array(self.dataset)

    def loadBatches(self):
        self.batches = []
        idx_list = np.arange(len(self.dataset))
        np.random.shuffle(idx_list)
        for i in range(0, len(idx_list), self.config['batch_size']):
            self.batches.append(self.dataset[idx_list[i: min(i + self.config['batch_size'], len(idx_list))]])

    def train(self):

        ax_loss = []

        for epoch in range(1, self.config['epoch'] + 1):

            total_loss = 0
            self.loadBatches()
            for batch in self.batches:
                x = [b[0] for b in batch]
                label = [b[1] for b in batch]

                x = Variable(torch.Tensor(x))
                label = Variable(torch.Tensor(label)).t()
                self.optimizer.zero_grad()
                y = self.model(x)
                loss = torch.sum((y.t() - label) ** 2)
                # loss = self.criterion(y.t(), label)
                total_loss += loss.data.numpy()[0]
                loss.backward()
                self.optimizer.step()

                if epoch % self.config['display_epoch'] == 0:
                    for _y, _label in zip(y.squeeze().data.numpy(), label.squeeze().data.numpy()):
                        print 'y={:6.2f} label={:6.2f}'.format(_y, _label)

            ax_loss.append(total_loss / len(self.dataset))
            if epoch % self.config['display_epoch'] == 0:
                print 'epoch={}, loss={:.2f}'.format(epoch, total_loss / len(self.dataset))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(1, len(ax_loss) + 1), ax_loss, 'r-', label='loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(loc='upper right')
        ax.grid()
        plt.show()

    def valid(self):
        self.model.train(False)

        y_label = []
        self.loadBatches()
        for batch in self.batches:
            x = [b[0] for b in batch]
            label = [b[1] for b in batch]

            x = Variable(torch.Tensor(x))
            label = Variable(torch.Tensor(label)).t()
            y = self.model(x)

            for _y, _label in zip(y.squeeze().data.numpy(), label.squeeze().data.numpy()):
                y_label.append([_y, _label])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot([x[1] for x in y_label], [x[0] for x in y_label], 'r+')
        ax.set_xlabel('label')
        ax.set_ylabel('predict_result')
        ax.grid()
        plt.show()

        self.model.train(True)



if __name__ == '__main__':


    config = {
        'epoch': 200,
        'display_epoch': 10,
        'generate_dataset': 10000,
        'batch_size': 32, 
        'lr': 0.001,
        'n': 2,
        'L': 10.,
        'hidden': 1
    }

    task = addtrain(config)
    task.train()
    task.valid()