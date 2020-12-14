import torch
import gym
import torch.nn as nn

from torch.distributions import Categorical
from ActorCritic import ActorNet
HIDDEN_SIZE = 256
agent = ActorNet(HIDDEN_SIZE)
agent.load_state_dict(torch.load('actor.pt'))
agent.eval()

env = gym.make("CartPole-v0").unwrapped
all_rewards = []
for i_episode in range(2):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        s = torch.from_numpy(s).float().unsqueeze(0)
        actions_probs = agent(s)
        probs = Categorical(actions_probs)
        a = probs.sample().item()
        s_, r, done, _ = env.step(a)
        s = s_
        ep_r += r
        if done:
            all_rewards.append(ep_r)
            break
print(all_rewards)





