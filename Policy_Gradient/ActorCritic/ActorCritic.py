import math
import random
from torch.distributions import Categorical

import gym
import numpy as py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


env = gym.make("CartPole-v0").unwrapped
# Hyper Parameter
N_STATE = env.observation_space.shape[0]
N_ACTION = env.action_space.n
LR = 3e-4
GAMMA = 0.9


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.layer1 = nn.Linear(N_STATE, 256)
        self.out = nn.Linear(256, N_ACTION)

    def forward(self, s):
        s = F.relu(self.layer1(s))
        s = self.out(s)
        return F.softmax(s, dim=1)


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.layer1 = nn.Linear(N_STATE, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, s):
        s = F.relu(self.layer1(s))
        value = self.out(s)
        return value


class ActorCritic(object):
    def __init__(self):
        self.actor_net, self.critic_net = ActorNet(), CriticNet()
        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=LR)
        self.save_logprobs = []
        self.learn_step = 0

    def select_action(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0)
        policy = self.actor_net(s)
        probs = Categorical(policy)
        action = probs.sample()
        self.save_logprobs.append(probs.log_prob(action))
        return action.item()

    def learn(self, s, a, r, s_):
        s = torch.from_numpy(s).float().unsqueeze(0)
        s_ = torch.from_numpy(s_).float().unsqueeze(0)
        self.learn_step += 0
        target_v = r + GAMMA * self.critic_net(s_).detach()
        predict_v = self.critic_net(s)
        critic_loss = self.critic_loss_func(target_v, predict_v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            td_error = target_v - predict_v
        actor_loss = -td_error*torch.cat(self.save_logprobs).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        del self.save_logprobs[:]
        return actor_loss.item()


if __name__ == '__main__':
    rewards = []
    losses = []
    ac = ActorCritic()
    for i_episode in range(10000):
        s = env.reset()
        ep_r = 0
        while True:
            # env.render()
            a = ac.select_action(s)
            s_, r, done, _ = env.step(a)

            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            ep_r += r
            losses.append(ac.learn(s, a, r, s_))

            if done:
                rewards.append(ep_r)
                print('Ep: ', i_episode,  # 输出该episode数
                      '| Ep_r: ', round(ep_r, 2))  # round()方法返回ep_r的小数点四舍五入到2个数字
                break
            s = s_


















