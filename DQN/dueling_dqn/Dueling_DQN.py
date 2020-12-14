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
GAMMA = 0.99
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
# plt.plot(x)
# plt.show()
# y = 1


def all_plot(learning_counter, rewards, losses, epsilon):
    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.title("rewards" + str(learning_counter))
    plt.plot(rewards)

    plt.subplot(132)
    plt.title("losses")
    plt.plot(losses)

    plt.subplot(133)
    plt.title("Epsilon")
    plt.plot(epsilon)

    plt.savefig("Dueling_DQN")

    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.feather = nn.Linear(N_STATES, 128)
        self.on1 = nn.Linear(128, 128)
        self.under1 = nn.Linear(128, 128)
        self.on2 = nn.Linear(128, 1)
        self.under2 = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.feather(x))
        value = F.relu(self.on1(x))
        value = self.on2(value)
        advantage = F.relu(self.under1(x))
        advantage = self.under2(advantage)
        return value + advantage - advantage.mean()





class Dueling_DQN(object):
    def __init__(self):
        self.target_net, self.predict_net = Net(), Net()
        self.buffer_counter = 0
        self.learning_counter = 0
        self.buffer = np.zeros((BUFFER_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s, learning_counter):
        s = torch.from_numpy(s).unsqueeze(0).float()
        actions_value = self.predict_net(s)
        if np.random.uniform() > 0.1:
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

    def learn(self):
        if self.learning_counter % TARGET_REPLACE_ITEM == 0:
            self.target_net.load_state_dict(self.predict_net.state_dict())
        self.learning_counter += 1

        sample_random = np.random.choice(BUFFER_CAPACITY,BATCH_SIZE )
        b = self.buffer[sample_random, :]
        b_s = torch.FloatTensor(b[:, :N_STATES])
        b_a = torch.LongTensor(b[:, N_STATES:N_STATES+1])
        b_r = torch.FloatTensor(b[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b[:, -N_STATES:])

        predict_q = self.predict_net(b_s).gather(1, b_a)
        target_a = torch.zeros((BATCH_SIZE, N_ACTIONS))
        for i in range(BATCH_SIZE):
            target_a[i] = self.predict_net(b_s_[i, :])
        target_a = torch.max(target_a, 1)[1].view(-1, 1)
        target_q = self.target_net(b_s_).gather(1, target_a).detach()
        target_q = target_q * GAMMA + b_r

        loss = self.loss_func(predict_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


dueing_dqn = Dueling_DQN()
all_rewards = []
all_losses = []
all_epsilon = []
for i_episode in range(10000):
    s = env.reset()
    ep_r = 0
    while True:
        a = dueing_dqn.choose_action(s, dueing_dqn.learning_counter)
        s_, r, done, _ = env.step(a)



        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        ep_r += r
        dueing_dqn.store_buffer(s, a, r, s_)
        if dueing_dqn.buffer_counter > BUFFER_CAPACITY:
            all_losses.append(dueing_dqn.learn())
            if done:
                print('Ep: ', i_episode,  # 输出该episode数
                      '| Ep_r: ', round(ep_r, 2), "|Epsilon: ", all_epsilon[dueing_dqn.learning_counter])
        if done:
            all_rewards.append(ep_r)
            break

        all_epsilon.append(adjust_epsilon(dueing_dqn.learning_counter))

        if dueing_dqn.learning_counter % 500 == 0 and dueing_dqn.learning_counter != 0:
            all_plot(dueing_dqn.learning_counter, all_rewards, all_losses, all_epsilon)

        s = s_


