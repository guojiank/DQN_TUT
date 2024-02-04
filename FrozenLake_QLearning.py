import collections
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = collections.deque(maxlen=5000)
        self.gamma = 0.88
        self.epsilon = 0.1
        self.q = QNetwork()
        self.target_q = QNetwork()
        self.q.load_state_dict(self.target_q.state_dict())
        self.loss = nn.SmoothL1Loss()
        self.opt = torch.optim.AdamW(self.q.parameters())
        self.step = 0

    def memory_append(self, s, a, r, next_s, t):
        self.memory.append((s, a, r, next_s, t))

    def update(self):
        if len(self.memory) <= 1000:
            return
        self.step += 1
        size_k = 64
        array = np.array(random.sample(self.memory, size_k))
        states = FloatTensor(array[:, 0]).reshape([-1, 1])
        actions = LongTensor(array[:, 1]).reshape([-1, 1])
        rewards = FloatTensor(array[:, 2]).reshape([-1, 1])
        next_states = FloatTensor(array[:, 3]).reshape([-1, 1])
        dones = FloatTensor(array[:, 4]).reshape([-1, 1])

        outputs = self.q(states).gather(1, actions)
        targets = rewards + self.gamma * (1 - dones) * self.target_q(next_states).max(1)[0].reshape([-1, 1])
        loss = self.loss(outputs, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.step % 20 == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return loss.item()

    def take_action(self, s):
        if np.random.uniform() <= self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            return self.q(FloatTensor([[s]])).argmax().item()

    def predict(self, s):
        return self.q(FloatTensor([[s]])).argmax().item()

    def learn(self):
        print(self.q(FloatTensor([[i] for i in range(16)])))
        env = gymnasium.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
        with trange(2000) as bar:
            for _ in bar:
                observation, _ = env.reset()
                for step in range(20):
                    action = self.take_action(observation)
                    observation_, reward, terminated, truncated, _ = env.step(action)
                    self.memory_append(observation, action, reward if observation != observation_ else -1,
                                       observation_, terminated)
                    observation = observation_
                    if terminated:
                        loss = self.update()
                        if loss is not None:
                            bar.set_description(f'loss: {loss}')
                        break

    def test(self):
        env = gymnasium.make('FrozenLake-v1', render_mode="human", is_slippery=False)
        print(self.q(FloatTensor([[i] for i in range(16)])))
        for episode in range(10):
            observation, _ = env.reset()
            for step in range(20):
                action = self.predict(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                observation = observation_
                if terminated or truncated:
                    break
            print(f'Episode finished after {episode}')


if __name__ == '__main__':
    agent = QAgent()
    agent.learn()
    agent.test()
