import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
from torch.distributions import Categorical
import os
import numpy as np
import matplotlib.pyplot as plt


UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MID_DIM = 128

env = gym.make('CartPole-v0')
N_STATE = env.observation_space.shape[0]
N_ACTION = env.action_space.n


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def push_and_pull(opt, gnet, lnet, done,  s_, b_s, b_r, b_a, gamma):
    if done is True:
        v_s_ = 0
    else:
        s_ = torch.from_numpy(s_).float().unsqueeze(0)
        _, v_s_ = lnet.forward(s_)
        v_s_ = v_s_.item()

    v_s = []
    for r in b_r[::-1]:
        s_v_ = r + v_s_ * GAMMA
        v_s.append(s_v_)
    v_s.reverse()
    v_s = torch.FloatTensor(v_s).view(-1, 1)
    b_a = torch.LongTensor(b_a)
    b_s = torch.FloatTensor(b_s).view(-1, N_STATE)
    loss = lnet.loss_func(b_s, b_a, v_s)
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp._grad
    opt.step()

    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(Net, self).__init__()
        self.ly1 = nn.Linear(state_dim, mid_dim)

        self.out_action = nn.Linear(mid_dim, action_dim)
        self.out_value = nn.Linear(mid_dim, 1)

    def forward(self, s):
        s = F.relu(self.ly1(s))
        advance_action = self.out_action(s)
        state_value = self.out_value(s)

        return F.softmax(advance_action, dim=1), state_value

    def choose_action(self, s):
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions_probs, value = self.forward(s)
        probs = Categorical(actions_probs)
        action = probs.sample()

        return action.item()

    def loss_func(self, s, a, s_v_):
        prob_action, values = self.forward(s)
        probs = Categorical(prob_action)
        td = s_v_ - values
        v_loss = td.pow(2)

        log_probs = probs.log_prob(a)
        a_loss = -td.detach() *log_probs

        total_loss = (v_loss + a_loss).mean()
        return total_loss




class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = "w%02i" % name
        self.gnet, self.opt = gnet, opt
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.lnet = Net(N_STATE, N_ACTION, MID_DIM)
        self.env = gym.make("CartPole-v0")

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            while True:
                a = self.lnet.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_r.append(r)
                buffer_a.append(a)
                ep_r += r

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.gnet, self.lnet, done, s_, buffer_s, buffer_r, buffer_a, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break

                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == '__main__':
    gnet = Net(N_STATE, N_ACTION, MID_DIM)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0), mp.Queue()

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [worker.start() for worker in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [worker.join() for worker in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()



