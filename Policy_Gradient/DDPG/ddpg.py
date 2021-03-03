import gym
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


# Hyper Parameters
MU_LR = 0.0005
Q_LR = 0.001
BUFFER_CAPACITY = 50000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.005


def plot(lossses, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title("rewards")
    plt.plot(range(20, 10000, 20), rewards)

    plt.subplot(132)
    plt.title("lossses")
    plt.plot(range(10000), lossses)

    plt.savefig('DDPG.PNG')
    plt.show()



class ReplayBuffer:
    def __init__(self, buffer_capacity):
        self.buffer = collections.deque(maxlen=buffer_capacity)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, next_s_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            next_s_lst.append(s_)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(next_s_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.out(x))*2
        return mu


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class DDPG(object):
    def __init__(self):
        self.mu, self.target_mu = ActorNet(), ActorNet()
        self.q, self.target_q = CriticNet(), CriticNet()
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=MU_LR)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=Q_LR)
        self.q_loss_func = nn.MSELoss()
        self.target_mu.load_state_dict(self.mu.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        self.learn_step = 0

    def train(self):
        s, a, r, s_, done = self.buffer.sample(BATCH_SIZE)
        target = r + self.target_q(s_, self.target_mu(s_)) * GAMMA * done
        q_loss = F.smooth_l1_loss(self.q(s, a), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        mu_loss = -self.q(s, self.mu(s)).mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()
        return q_loss.item()

    def soft_update(self):
        for param_target, parm in zip(self.target_mu.parameters(), self.mu.parameters()):
            param_target.data.copy_(param_target.data * (1 - TAU) + TAU * parm.data)

        for param_target, parm in zip(self.target_q.parameters(), self.q.parameters()):
            param_target.data.copy_(param_target.data * (1 - TAU) + TAU * parm.data)


if __name__ == '__main__':
    ddpg = DDPG()
    env = gym.make('Pendulum-v0')
    rewards = []
    losses = []
    score = 0
    print_interval = 20
    for i_episode in range(10000):
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = ddpg.mu(torch.from_numpy(s).float().unsqueeze(0)).item()
            a = a + ddpg.ou_noise()[0]
            s_, r, done, info = env.step([a])
            ddpg.buffer.put((s, a, r/100.0, s_, done))
            score += r
            s = s_
        if ddpg.buffer.size() > 2000:
            for i in range(10):
                losses.append(ddpg.train())
                ddpg.soft_update()
                ddpg.learn_step += 1

        if i_episode % print_interval == 0 and i_episode != 0:
            print("# of episode :{}, avg score : {:.1f}".format(i_episode, score / print_interval))
            score = 0
    env.close()
    torch.save(ddpg.mu.state_dict(), 'hello.pt')





























