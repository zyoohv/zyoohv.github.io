#! /usr/bin/python3


'''
nature DQN:
    1. experinence replay
    2. target network
'''

import gym
import copy
import numpy as np
import matplotlib.pyplot as plt
from itertools import count


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class net(nn.Module):
    def __init__(self, **args):
        super(net, self).__init__()

        self.linear1 = nn.Linear(args['n_states'], args['n_hidden'])
        # self.linear2 = nn.Linear(args['n_hidden'], args['n_actions'])
        self.linear2 = nn.Linear(args['n_hidden'], args['n_hidden'])
        self.linear3 = nn.Linear(args['n_hidden'], args['n_actions'])

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class datasetutil():
    def __init__(self, **args):
        '''
        buff_size = 512
        '''
        self.config = args

        # format [[origin_state], action, reward, [next_state]]
        self.dataset_buff = []

        self.is_ready = False

    @property
    def isready(self):
        return self.is_ready
    

    def add_action(self, origin_state, action, reward, next_state):
        if len(self.dataset_buff) >= self.config['buff_size']:
            del self.dataset_buff[0]

        self.dataset_buff.append([origin_state, action, reward, next_state])
        if self.is_ready is False and \
            batch_size == len(self.dataset_buff):
            self.is_ready = True

    def sample(self, batch_size=32):
        sample_idx = np.random.choice(len(self.dataset_buff), 
            size=batch_size)

        origin_state = np.array([self.dataset_buff[i][0] for i in sample_idx])
        action = np.array([[self.dataset_buff[i][1]] for i in sample_idx])
        reward = np.array([[self.dataset_buff[i][2]] for i in sample_idx])
        next_state = np.array([self.dataset_buff[i][3] for i in sample_idx])

        return origin_state, action, reward, next_state


class qlearning():
    def __init__(self, **args):
        '''
        example:
            n_states = 4
            n_hidden = 128
            n_actions = 4

            learning_rate = 0.01
            epsilon = 0.9
            gamma = 0.9
        '''

        self.config = args

        self.eval_net = net(n_states=self.config['n_states'], 
            n_hidden=self.config['n_hidden'], 
            n_actions=self.config['n_actions'])

        self.target_net = net(n_states=self.config['n_states'], 
            n_hidden=self.config['n_hidden'], 
            n_actions=self.config['n_actions'])
        self.target_net.train(False)
        
        self.loss_func = nn.MSELoss()
        # self.optimizer = optim.SGD(self.eval_net.parameters(), 
        #     lr=self.config['learning_rate'], momentum=0.9)
        self.optimizer = optim.Adam(self.eval_net.parameters(),
            lr=self.config['learning_rate'])

    def select_action(self, state, epsilon_greedy=True):
        self.eval_net.train(False)

        if epsilon_greedy is False or \
            np.random.uniform() < self.config['epsilon']:
            state = Variable(torch.Tensor(state))
            action_score = self.eval_net(state).squeeze().data.numpy()
            action = np.argmax(action_score)
        else:
            action = np.random.randint(0, self.config['n_actions'])

        return action

    def train(self, origin_state, action, reward, next_state):
        # Nature DQN
        self.eval_net.train(True)
        self.optimizer.zero_grad()

        eval_score = self.eval_net(Variable(torch.Tensor(origin_state)))
        next_score = self.target_net(Variable(
            torch.Tensor(next_state))).data.numpy()

        # Double DQN
        # self.eval_net.train(False)

        # self.optimizer.zero_grad()
        # eval_next_score = self.eval_net(Variable(
        #     torch.Tensor(next_state))).data.numpy()
        # next_score = self.target_net(Variable(
        #     torch.Tensor(next_state))).data.numpy()

        # self.eval_net.train(True)
        # eval_score = self.eval_net(Variable(torch.Tensor(origin_state)))

        target_score = copy.deepcopy(eval_score.data.numpy())
        for i in range(len(target_score)):
            # the core update function here.
            # print(np.argmax(eval_next_score[i]))

            target_score[i][action[i][0]] = reward[i][0] + \
                self.config['gamma'] * np.max(next_score[i])


            # target_score[i][action[i][0]] = reward[i][0] + \
            #     self.config['gamma'] * next_score[i][ \
            #         np.argmax(eval_next_score[i])]

        target_score = Variable(torch.Tensor(target_score))
        # print('eval_score=\n', eval_score)
        # print('target_score=\n', target_score)
        # print("############################################")
        loss = self.loss_func(eval_score, target_score)
        loss.backward()
        self.optimizer.step()

        return float(loss.mean().data.numpy())

    def copy_parameters(self):

        eval_state_dict = self.eval_net.state_dict()
        target_state_dict = self.target_net.state_dict()

        for name, param in eval_state_dict.items():
            target_state_dict[name].copy_(param)


def main():
    train_score_his = []
    test_score_his = []
    loss_his = []
    
    for step in range(run_epoch):

        if step % 100 == 0:
            print('epoch=', step)

        states_action_log = []
        states = env.reset()

        while True:
            action = ql.select_action(states)
            # ndarray, float, boolean, dict --- int
            states_, reward, done, _ = env.step(action)
            states_action_log.append([states, action, reward, states_])
            states = states_
            if done:
                break

        train_score_his.append(len(states_action_log))

        states_action_log.reverse()
        for i in range(1, len(states_action_log)):
            states_action_log[i][2] += gamma * states_action_log[i - 1][2]
        # reward_norm = np.array([states_action_log[i][2] for i in  
        #     range(len(states_action_log))])
        # reward_norm = (reward_norm - np.mean(reward_norm)) / \
        #     (np.std(reward_norm))
        # for i in range(len(states_action_log)):
        #     states_action_log[i][2] = reward_norm[i]

        for origin_state, action, reward, next_state in states_action_log:
            dt.add_action(origin_state, action, reward, next_state)

        if dt.isready:
            for _ in range(buff_size // batch_size):
                origin_state, action, reward, next_state = dt.sample(
                    batch_size=batch_size)
                loss = ql.train(origin_state, action, reward, next_state)
                loss_his.append(loss)

        if step % update_epoch == 0:
            ql.copy_parameters()

        if step % test_epoch == 0:
            states = env.reset()
            test_score_his.append(0)

            while True:
                test_score_his[-1] += 1
                action = ql.select_action(states, epsilon_greedy=False)
                states, reward, done, _ = env.step(action)
                # env.render()
                if done:
                    break


    # print('loss history=', loss_his)

    loss_fig = plt.figure()
    loss_ax = loss_fig.add_subplot(111)
    loss_ax.plot(range(len(loss_his)), loss_his, 'k-')
    loss_ax.set_title('loss')

    train_score_fig = plt.figure()
    train_score_ax = train_score_fig.add_subplot(111)
    train_score_ax.plot(range(len(train_score_his)), train_score_his, 'k-')
    train_score_ax.set_title('train_score')

    test_score_fig = plt.figure()
    test_score_ax = test_score_fig.add_subplot(111)
    test_score_ax.plot(range(len(test_score_his)), test_score_his, 'k-')
    test_score_ax.set_title('test_score')

    plt.show()


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    
    states_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    epsilon = 0.9
    gamma = 0.9
    learning_rate = 0.001
    n_hidden = 128
    buff_size = 512
    batch_size = 32
    run_epoch = 500
    test_epoch = 1
    update_epoch = 1

    ql = qlearning(n_states=states_size,
                   n_hidden=n_hidden,
                   n_actions=action_size,
                   learning_rate=learning_rate,
                   epsilon=epsilon,
                   gamma=gamma
                   )
    dt = datasetutil(buff_size=buff_size)
    
    main()