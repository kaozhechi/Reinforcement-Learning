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


def adjust_epsilon(learn_counter):
    epsilon = EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1 * learn_counter/ EPSILON_DACY)
    return epsilon


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(N_STATES, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_advantage = nn.Linear(128, N_ACTIONS)

    def forward(self, s):
        s = nn.ReLU(self.fc(s))
        s_value = nn.ReLU(self.fc_value(s))
        s_advantage = nn.ReLU(self.fc_advantage(s))
        s_q = s_value + s_advantage
        return s_q


