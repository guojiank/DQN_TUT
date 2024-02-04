import collections
import math
import random

import gymnasium
import numpy as np
import torch
from torch import nn, FloatTensor, LongTensor
from tqdm import *


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.8
        self.q = collections.defaultdict(lambda: np.zeros(4))

    def train(self):
        print('--------------------- begin train ---------------------')
        env = gymnasium.make('FrozenLake-v1', max_episode_steps=200, render_mode="rgb_array", is_slippery=False)
        for episode in range(200):
            observation, _ = env.reset()
            for step in range(100):
                env.render()
                action = self.sample(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                self.update(observation, action, reward, observation_, terminated)
                observation = observation_
                if terminated:
                    print(f'episode {episode} finished after:{step} ')
                    for item in list(self.q.items()):
                        print(item)
                    break

    def test(self):
        print('--------------------- begin test ---------------------')
        env = gymnasium.make('FrozenLake-v1', render_mode="human", is_slippery=False)
        for episode in range(10):
            observation, _ = env.reset()
            for step in range(100):
                env.render()
                action = self.sample(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                observation = observation_
                if terminated:
                    print(f'episode {episode} finished after:{step} ')
                    break

    def sample(self, s):
        return random.choice(np.argwhere(self.q[s] == np.max(self.q[s])))[0]

    def update(self, s, a, r, next_s, terminated):
        if s == next_s:
            r = -1
        if terminated and r != 1:
            r = -1
        self.q[s][a] = float(r) + self.gamma * max(self.q[next_s])
        pass


class QNetwork(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.model(x)


class QAgent(object):

    def __init__(self):
        self.memory = collections.deque([], maxlen=5000)
        self.gamma = 0.88
        self.q = QNetwork()
        self.target_q = QNetwork()
        self.q.load_state_dict(self.target_q.state_dict())
        self.loss = nn.SmoothL1Loss()
        self.opt = torch.optim.AdamW(self.q.parameters())
        self.count = 0

    def append(self, s, a, r, next_s):
        self.memory.append(
            (FloatTensor([s]), LongTensor([a]), FloatTensor([r]), FloatTensor([next_s]) if next_s else None)
        )

    def update(self):
        if len(self.memory) <= 128:
            return
        size_k = 128
        batch = list(zip(*random.sample(self.memory, size_k)))
        states = torch.cat(batch[0]).reshape([-1, 1])
        actions = torch.cat(batch[1]).reshape([-1, 1])
        rewards = torch.cat(batch[2]).reshape([-1, 1])
        masks = torch.BoolTensor([[i is not None] for i in batch[3]])
        not_none_next_states = torch.cat([i for i in batch[3] if i is not None]).reshape([-1, 1])

        next_states_values = torch.zeros(128).reshape(-1, 1)
        next_states_values[masks] = self.target_q(not_none_next_states).max(1)[0]

        outputs = self.q(states).gather(1, actions)
        targets = rewards + self.gamma * next_states_values
        loss = self.loss(outputs, targets)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        q_dict = self.q.state_dict()
        target_q_dict = self.target_q.state_dict()

        for k in target_q_dict.keys():
            target_q_dict[k] = q_dict[k] * 0.005 + 0.995 * target_q_dict[k]

        self.target_q.load_state_dict(target_q_dict)

    def sample(self, s):
        self.count += 1
        e = 0.05 + 0.85 * math.exp(-1.0 * self.count / 1000)
        if random.random() < e:
            return random.choice([0, 1, 2, 3])
        else:
            return self.action(s)

    def action(self, s):
        return self.q(FloatTensor([[s]])).argmax().item()

    def learn(self):
        env = gymnasium.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
        with trange(500) as bar:
            for _ in bar:
                observation, _ = env.reset()
                for step in range(20):
                    action = self.sample(observation)
                    observation_, reward, terminated, truncated, _ = env.step(action)
                    self.append(observation, action, -1 if observation_ == observation else reward,
                                None if terminated else observation_)
                    observation = observation_
                    self.update()
                    if terminated:
                        break

    def test(self):
        env = gymnasium.make('FrozenLake-v1', render_mode="human", is_slippery=False)
        for episode in range(10):
            observation, _ = env.reset()
            for step in range(20):
                action = self.action(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                observation = observation_
                if terminated or truncated:
                    break
            print(f'Episode finished after {episode}')


if __name__ == '__main__':
    agent = QAgent()
    agent.learn()
    agent.test()
