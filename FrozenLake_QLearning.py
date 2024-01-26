import collections
import gymnasium
import numpy as np
import random

import torch
from torch import nn


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
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)


class QAgent(object):

    def __init__(self):
        self.memory = collections.deque(maxlen=1000)
        self.lr = 0.001
        self.q = QNetwork()
        self.target_q = QNetwork()

    def memory_update(self, s, a, r, next_s, terminated):
        self.memory.append((s, a, r, next_s, terminated))

    def update_q(self):
        pass


if __name__ == '__main__':
    agent = QLearningAgent()
    agent.train()
    agent.test()
