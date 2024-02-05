from itertools import count

import gymnasium as gym
from tqdm import trange

import math
from collections import deque
from random import random, sample

import gymnasium
import torch
from torch import FloatTensor, optim, nn, LongTensor


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x)


class Memory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(*args))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)


class Agent:

    def __init__(self, env: gymnasium.Env, memory_capacity=10000, batch_size=64, gamma=0.99, tun=0.005):
        self.env = env
        self.memory = Memory(memory_capacity)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.policy_net = Network(self.input_size, self.output_size)
        self.target_net = Network(self.input_size, self.output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters())
        self.batch_size = batch_size
        self.gamma = gamma
        self.tun = tun
        self.count = 0
        pass

    def push(self, s, a, r, s_):
        self.memory.push(
            (FloatTensor([s]), LongTensor([a]), FloatTensor([r]), FloatTensor([s_]) if s_ is not None else None)
        )

    def action(self, s):
        e = 0.05 + 0.90 * math.exp(-1.0 * self.count / 1000)
        self.count += 1
        if random() > e:
            return self.env.action_space.sample()
        else:
            return self.policy_net(FloatTensor([s])).argmax().item()

    def choose_action(self, s):
        return self.policy_net(FloatTensor([s])).argmax().item()

    def learn(self):
        if self.memory.size() < self.batch_size:
            return
        batch = list(zip(*self.memory.sample(self.batch_size)))
        batch_state = torch.cat(batch[0]).reshape(-1, self.input_size)
        batch_action = torch.cat(batch[1]).reshape(-1, 1)
        batch_reward = torch.cat(batch[2]).reshape(-1, 1)

        next_not_null_batch_state = torch.cat([s for s in batch[3] if s is not None]).reshape(-1, self.input_size)
        next_not_null_mask = torch.BoolTensor([s is not None for s in batch[3]]).reshape(-1, 1)

        batch_action_values = self.policy_net(batch_state).gather(1, batch_action)

        batch_next_action_values = torch.zeros(self.batch_size).reshape(-1, 1)
        batch_next_action_values[next_not_null_mask] = self.target_net(next_not_null_batch_state).max(1).values

        batch_target_values = batch_reward + self.gamma * batch_next_action_values

        loss = self.loss_fn(batch_action_values, batch_target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        policy_dict = self.policy_net.state_dict()
        target_dict = self.target_net.state_dict()

        for key in policy_dict.keys():
            target_dict[key] = policy_dict[key] * self.tun + target_dict[key] * (1 - self.tun)

        self.target_net.load_state_dict(target_dict)


if __name__ == '__main__':
    with gym.make("LunarLander-v2", render_mode="rgb_array") as env, trange(300) as bar:
        observation, _ = env.reset(seed=42)
        agent = Agent(env)
        for episode in bar:
            for _ in count():
                action = agent.action(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    reward = -1000
                agent.push(observation, action, reward, None if terminated else observation_)
                agent.learn()
                observation = observation_
                if terminated or truncated:
                    observation, info = env.reset()
                    break

    with gym.make("LunarLander-v2", render_mode="human") as env, trange(10) as bar:
        observation, _ = env.reset(seed=42)
        for episode in bar:
            for _ in count():
                action = agent.choose_action(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                observation = observation_
                if terminated or truncated:
                    observation, info = env.reset()
                    break
