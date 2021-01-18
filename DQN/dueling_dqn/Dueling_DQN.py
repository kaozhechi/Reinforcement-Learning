import math,random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

env = gym.make('CartPole-v0').unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
LR = 0.01
EPSILON = 0.9
EPISODE = 200
GAMMA = 0.99
BUFFER_CAPACITY = 2000
TARGET_REPLACE_ITEM = 100
BATCH_SIZE = 32


def all_plt(learning_counter, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title("rewards" + str(learning_counter))
    plt.plot(rewards)

    plt.subplot(132)
    plt.title("losses")
    plt.plot(losses)

    plt.show()



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(N_STATES, 128)
        self.on2 = nn.Linear(128, 128)
        self.under2 = nn.Linear(128, 128)
        self.on3 = nn.Linear(128, 1)
        self.under3 = nn.Linear(128, N_ACTIONS)

    def forward(self, s):
        s = F.relu(self.layer1(s))
        value = F.relu(self.on2(s))
        value = self.on3(value)
        advantage = F.relu(self.under2(s))
        advantage = self.under3(advantage)
        return value + advantage - advantage.mean()


class Dueling_DQN(object):
    def __init__(self):
        super(Dueling_DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.learning_counter = 0
        self.buffer_counter = 0
        self.buffer = np.zeros((BUFFER_CAPACITY, N_STATES*2+2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0)
        if np.random.uniform() > EPSILON:
            action = np.random.randint(0, N_ACTIONS)
        else:
            actions_value = self.eval_net(s)
            action = torch.max(actions_value, 1)[1].numpy()
            action = action.item()
        return action

    def store_buffer(self, s, a, r, s_):
        transition = np.hstack((s, (a, r), s_))
        index = self.buffer_counter % BUFFER_CAPACITY
        self.buffer[index, :] = transition
        self.buffer_counter += 1

    def learn(self):
        if self.learning_counter % TARGET_REPLACE_ITEM == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_counter += 1

        sample_random = np.random.choice(BUFFER_CAPACITY, BATCH_SIZE)
        b = self.buffer[sample_random, :]
        b_s = torch.FloatTensor(b[:, :N_STATES])
        b_a = torch.LongTensor(b[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b[:, -N_STATES:])

        predict_q = self.eval_net(b_s).gather(1, b_a)
        target_a = self.eval_net(b_s_)
        target_a = torch.max(target_a, 1)[1].view(-1, 1)
        target_q = self.target_net(b_s_).gather(1, target_a).detach()
        target_q = target_q * GAMMA + b_r

        loss = self.loss_func(target_q, predict_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


if __name__ == '__main__':
    d3qn = Dueling_DQN()
    rewards = []
    losses = []
    for i_episode in range(10000):
        s = env.reset()
        ep_r = 0
        while True:
            a = d3qn.choose_action(s)
            s_, r, done, _ = env.step(a)

            # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            ep_r += r
            d3qn.store_buffer(s, a, r, s_)

            if d3qn.buffer_counter >= BUFFER_CAPACITY:
                losses.append(d3qn.learn())
                if done:
                    print('Ep: ', i_episode,  # 输出该episode数
                          '| Ep_r: ', round(ep_r, 2))
                    rewards.append(ep_r)
            if d3qn.learning_counter % 10000 == 0 and d3qn.learning_counter != 0:
                all_plt(d3qn.learning_counter, rewards, losses)
            if done:
                break
            s = s_












