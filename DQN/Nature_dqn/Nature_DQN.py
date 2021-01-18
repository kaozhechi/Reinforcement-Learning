import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym

import math
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
GAMMA = 0.9
EPSILON = 0.9
MEMORY_CAPACITY = 2000
TARGET_REPLACE_ITEM = 100

env = gym.make('CartPole-v0').unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n


def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)

    plt.subplot(132)
    plt.title("losses")
    plt.plot(losses)
    plt.savefig('Nature_DQN.PNG')
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)
        self.out = nn.Linear(50, N_ACTIONS)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        actions_value = self.out(s)
        return actions_value


class NatureDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):

        if np.random.uniform() > EPSILON:
            action = np.random.randint(0, N_ACTIONS)
        else:
            s = torch.from_numpy(s).float().unsqueeze(0)
            actions_value = self.eval_net(s)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        return action

    def store_buffer(self, s, a, r, s_):
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
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1: N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == '__main__':
    dqn = NatureDQN()
    print('\nCollecting experience...')
    rewards = []
    losses = []
    for i_episode in range(400):
        ep_r = 0
        s = env.reset()

        while True:
            # env.render()
            a = dqn.choose_action(s)
            s_, r, done, _ = env.step(a)

            # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_buffer(s, a, r, s_)
            ep_r += r

            if dqn.memory_counter >= MEMORY_CAPACITY:
                losses.append(dqn.learn())
                if done:
                    print('Ep: ', i_episode,  # 输出该episode数
                          '| Ep_r: ', round(ep_r, 2))  # round()方法返回ep_r的小数点四舍五入到2个数字

            if dqn.learn_step_counter % 10000 == 0 and dqn.learn_step_counter != 0:
                plot(dqn.learn_step_counter, rewards, losses)

            if done:
                rewards.append(ep_r)
                break

            s = s_

            if sum(rewards[-5:]) >= 3000:
                print("ok")
                torch.save(dqn.eval_net.state_dict(), 'actor.pt')










