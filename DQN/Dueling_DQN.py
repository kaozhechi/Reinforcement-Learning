import math,random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

env = gym.make("CartPole-v0")
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
EPSILON_START = 1.0
EPSILON_FINAL = 0.0
EPSILON_DACY = 500
GAMMA = 0.9
LR = 0.01
BUFFER_CAPACITY = 2000
TARGET_REPLACE_ITEM = 100
BATCH_SIZE = 32


def adjust_epsilon(learn_counter):
    epsilon = EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1 * learn_counter/ EPSILON_DACY)
    return epsilon

# x = []
# for i in range(10000):
#     x.append(adjust_epsilon(i))
# # plt.plot(x)
# # plt.show()
# # y = 1

def all_plot(learning_counter, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title("rewards" + learning_counter)
    plt.plot(rewards)
    plt.subplot(132)
    plt.title("losses")
    plt.plot(losses)
    plt.savefig("Dueling_DQN")
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,N_ACTIONS)
        )


def forward(self, x):
    x = self.feature(x)
    advantage = self.advantage(x)
    value = self.value(x)
    return value + advantage - advantage.mean()


class Dueling_DQN(object):
    def __init__(self):
        self.target_net, self.predict_net = Net(), Net()
        self.buffer_counter = 0
        self.leaning_counter = 0
        self.buffer = np.zeros((BUFFER_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s, learning_counter):
        s = torch.from_numpy(s).unsqueeze(0).float()
        actions_value = self.predict_net(s)
        if np.random.uniform() > adjust_epsilon(learning_counter):
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action.item()
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_buffer(self, s, a, r, s_):
        index = self.buffer_counter % BUFFER_CAPACITY
        transition = np.hstack((s, [a, r], s_))
        self.buffer[index, :] = transition
        self.buffer_counter += 1

    def learning(self):
        if self.leaning_counter % TARGET_REPLACE_ITEM == 0:
            self.target_net.load_state_dict(self.predict_net.state_dict())
        self.leaning_counter += 1

        sample_random = np.random.choice(BUFFER_CAPACITY,BATCH_SIZE )
        b = self.buffer[sample_random, :]
        b_s = torch.FloatTensor(b[:, :N_STATES])
        b_a = torch.LongTensor(b[:, N_STATES:N_STATES+1])
        b_r = torch.FloatTensor(b[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b[:, -N_STATES:])

        predict_q = self.predict_net(b_s).gather(1, b_a)
        target_a = []
        for i in range(BATCH_SIZE):
            target_a.append(b_s_[i])
        target_a = torch.FloatTensor(target_a).unsqueeze(0)
        target_q = self.target_net(b_s_).gather(1, target_a)
        target_q = target_q * GAMMA + b_r

        loss = self.loss_func(predict_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


dueing_dqn = Dueling_DQN()
all_rewards = []
all_losses =[]
for i_episode in range(10000):
    s = env.reset()
    ep_r = 0
    while True:
        a = dueing_dqn.choose_action(s, dueing_dqn.leaning_counter)
        s_, r, done, _ = env.step(a)
        ep_r += r

        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        if dueing_dqn.buffer_counter >= BUFFER_CAPACITY:
            all_losses.append(dueing_dqn.learning())

        if done:
            all_rewards.append(ep_r)

        s = s_
        if dueing_dqn.buffer_counter > BUFFER_CAPACITY:
            all_losses.append(dueing_dqn.learning())
            if done:
                print('Ep: ', i_episode,  # 输出该episode数
                      '| Ep_r: ', round(ep_r, 2))  # round()方法返回ep_r的小数点四舍五入到2个数字





