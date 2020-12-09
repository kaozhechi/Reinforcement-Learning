import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym

import math
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32  # 批处理时的样本数量
LR = 0.01  # 学习率，步长
GAMMA = 0.9  # 折扣
EPSILON_START = 1
EPSILON_FINAL = 0
EPSILON_DECAY = 500
MEMORY_CAPACITY = 2000  # 经验池的大小
TARGET_REPLACE_ITEM = 100  # 建立了单独的目标网络 来处理TD误差 来解决训练的不稳定性
env = gym.make('CartPole-v0').unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n


def adjust_epsilon(learn_counter):
    epsilon = EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL)*math.exp(-1. * learn_counter / EPSILON_DECAY)
    return epsilon


def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig("Double_DQN.PNG")
    plt.show()




# 定义neural network 框架
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)

        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() > epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITEM == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


dqn = DQN()
print('\nCollecting experience...')
all_rewards = []
all_loss = []
for i_episode in range(10000):
    ep_r = 0
    s = env.reset()

    while True:
        # env.render()
        a = dqn.choose_action(s,adjust_epsilon(dqn.learn_step_counter))
        s_, r, done, info = env.step(a)

        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)
        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            all_loss.append(dqn.learn())
            if done:
                print('Ep: ', i_episode,  # 输出该episode数
                      '| Ep_r: ', round(ep_r, 2))  # round()方法返回ep_r的小数点四舍五入到2个数字
        if dqn.learn_step_counter % 10000 == 0 and dqn.learn_step_counter != 0:
            plot(dqn.learn_step_counter, all_rewards, all_loss)
        if done:  # 如果满足终止条件
            all_rewards.append(ep_r)
            break  # 该episode结束
        s = s_
        # if dqn.learn_step_counter % 200 == 0:







