import math
import random
from torch.distributions import Categorical

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import matplotlib.pyplot as plt

env = gym.make("CartPole-v0").unwrapped
# Hyper Parameters
N_STATE = env.observation_space.shape[0]
N_ACTION = env.action_space.n
LR = 3e-4
GAMMA = 0.9
HIDDEN_SIZE = 256


def plot_all(rewards, loss):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title("rewards")
    plt.plot(rewards)

    plt.subplot(132)
    plt.title("Loss")
    plt.plot(loss)

    plt.show()


class ActorNet(nn.Module):
    def __init__(self, hidden_size):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(N_STATE, hidden_size)
        self.out = nn.Linear(hidden_size, N_ACTION)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.out(s)
        return F.softmax(s, dim=1)


class CriticNet(nn.Module):
    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(N_STATE, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        value = self.out(s)
        return value


class ActorCritic(object):
    def __init__(self):
        self.actor_net, self.critic_net = ActorNet(HIDDEN_SIZE), CriticNet(HIDDEN_SIZE)
        self.critic_loss_func = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=LR)
        self.save_logprobs = []

    def select_action(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions_prob = self.actor_net(s)
        probs = Categorical(actions_prob)  # 参数actions_prob为标准的类别分布
        action = probs.sample()
        self.save_logprobs.append(probs.log_prob(action))
        return action.item()

    def learn(self, s, a, r, s_):
        s = torch.from_numpy(s).float().unsqueeze(0)
        s_ = torch.from_numpy(s_).float().unsqueeze(0)

        s_value = self.critic_net(s_).detach()
        target_v = r + GAMMA * s_value
        predict_v = self.critic_net(s)
        cirtic_loss = self.critic_loss_func(predict_v, target_v)
        self.critic_optimizer.zero_grad()
        cirtic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            td_error = (target_v - predict_v).item()
        actor_loss = - td_error * torch.cat(self.save_logprobs).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        del self.save_logprobs[:]
        return actor_loss.item()

    def env_train(self, episode):
        all_rewards = []
        all_loss = []
        learn_step = 0
        for i in range(episode):
            s = env.reset()
            ep_reward = 0
            while True:
                a = self.select_action(s)
                s_, r, done, _ = env.step(a)

                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                ep_reward += r
                all_loss.append(self.learn(s, a, r, s_))
                learn_step += 1
                if done:
                    all_rewards.append(ep_reward)
                    break
                if learn_step % 10000 == 0 and learn_step != 0:
                    plot_all(all_rewards, all_loss)
                s = s_


agent = ActorCritic()
agent.env_train(400)
