import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
# Hyper Parameters
BATCH_SIZE = 32    # 批处理时的样本数量
LR = 0.01          # 学习率，步长
GAMMA = 0.9        # 折扣
EPSILON = 0.9      # 伊普西隆-greedy
MEMORY_CAPACITY = 2000  # 经验池的大小
TARGET_REPLACE_ITEM = 100  # 建立了单独的目标网络 来处理TD误差 来解决训练的不稳定性
env = gym.make('CartPole-v0').unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n


# 定义neural network 框架
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 120)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(120, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)

        return actions_value


class DDQN(object):
    def __init__(self):
        self.target_net, self.predict_net = Net(), Net()
        self.memory_counter = 0
        self.learn_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.Tensor(state).unsqueeze(0)
        if np.random.uniform() < EPSILON:
            actions_value = self.predict_net.forward(state)
            # action = torch.argmax(actions_value, dim=1).data.numpy()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % MEMORY_CAPACITY
        transition = np.hstack((s, [a, r], s_))
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_counter % TARGET_REPLACE_ITEM == 0:
            self.target_net.load_state_dict(self.predict_net.state_dict())
        self.learn_counter += 1
        batch = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[batch, :]
        b_s = torch.from_numpy(b_memory[:, :N_STATES]).float()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.from_numpy(b_memory[:, N_STATES+1:N_STATES+2]).float()
        b_s_ = torch.from_numpy(b_memory[:, -N_STATES:]).float()
        target_a = torch.zeros(BATCH_SIZE, N_ACTIONS)
        for index in range(b_s_.shape[0]):
            x = b_s_[index, :].unsqueeze(0)
            target_a[index, :] = self.predict_net(b_s_[index, :].unsqueeze(0))
        target_a = torch.max(target_a, 1)[1].unsqueeze(0)
        target_a = target_a.view(32, 1)
        q_predict = self.predict_net(b_s).gather(1, b_a)
        target_actions_value = self.target_net(b_s_).detach()
        q_target = GAMMA * target_actions_value.gather(1, target_a) + b_r
        loss = self.loss_func(q_predict, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

ddqn = DDQN()
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        # env.render()
        a = ddqn.choose_action(s)
        s_, r, done, _ = env.step(a)


        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        ddqn.store_transition(s, a, r, s_)
        ep_r += r
        if ddqn.memory_counter > MEMORY_CAPACITY:
            ddqn.learn()
            if done:
                print('Ep: ', i_episode,  # 输出该episode数
                      '| Ep_r: ', round(ep_r, 2))  # round()方法返回ep_r的小数点四舍五入到2个数字

        if done:  # 如果满足终止条件
            break  # 该episode结束
        s = s_










