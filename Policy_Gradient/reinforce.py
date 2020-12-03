import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F  # 一些relu等激活函数
import torch.optim as optim  # 用来调用adam
from torch.distributions import Categorical

GAMMA = 0.99
LR = 1e-2
torch.manual_seed(543)

env = gym.make('CartPole-v1')
N_STATE = env.observation_space.shape[0]
N_ACTION = env.action_space.n
env.seed(543)
EPS = np.finfo(np.float64).eps.item()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATE, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.out = nn.Linear(128, N_ACTION)

    def forward(self,state):
        state = self.fc1(state)
        state = self.dropout(state)
        state = F.relu(state)
        state = self.out(state)
        return F.softmax(state, dim=1)


class Reinforce(object):
    def __init__(self):
        self.net = Net()
        self.save_log_probs = []
        self.rewards = []
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

    def select_action(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions_prob = self.net(s)
        probs = Categorical(actions_prob)  # 参数actions_prob为标准的类别分布
        action = probs.sample()
        self.save_log_probs.append(probs.log_prob(action))
        return action.item()

    def learn(self):
        loss = []
        returns = []
        G = 0
        for R in self.rewards[::-1]:
            G = R + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)
        for log_prob, g in zip(self.save_log_probs, returns):
            loss.append(-log_prob*g)

        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()

        del self.save_log_probs[:]
        del self.rewards[:]


reinforce = Reinforce()
running_reward = 10
for i_episode in count(1):
    s = env.reset()
    ep_reward = 0
    for t in range(1, 10000):
        a = reinforce.select_action(s)
        s_, reward, done, _ = env.step(a)
        reinforce.rewards.append(reward)
        ep_reward += reward
        s = s_
        if done:
            break
    running_reward = ep_reward*0.05 + running_reward*0.95
    reinforce.learn()
    if i_episode % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
        i_episode, ep_reward, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        torch.save(reinforce.net.state_dict(), 'hello.pt')
        break































