import math
from collections import deque
import random

import gymnasium
import numpy as np
import torch.optim
from torch import nn, FloatTensor
from tqdm import trange


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)


MEMORY_SIZE = 10000


class Memory:
    def __init__(self):
        self.q = deque([], maxlen=MEMORY_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.q.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.q, batch_size)

    def size(self):
        return len(self.q)


class Agent:
    def __init__(self):
        self.q_network = Net()
        self.target_q_network = Net()
        self.loss_fn = nn.SmoothL1Loss()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=0.0001)
        self.counter = 0
        pass

    def choose_action(self, state):
        e = 0.05 + (0.9 - 0.05) * math.exp(-1. * self.counter / 1000)
        self.counter += 1
        if random.random() < e:
            return random.choice([0, 1])
        else:
            return self.action(state)

    def action(self, state):
        return self.q_network(FloatTensor(np.array([state]))).argmax().item()

    def train(self, m: Memory):
        if m.size() < MEMORY_SIZE:
            return
        arr = np.array(m.sample(128))

        b_s = FloatTensor(arr[:, 1]).reshape(-1, 4)
        b_a = FloatTensor(arr[:, 2]).reshape(-1, 1)
        b_r = FloatTensor(arr[:, 3]).reshape(-1, 1)
        b_s_ = FloatTensor(arr[:, 4]).reshape(-1, 4)
        b_dones = FloatTensor(arr[:, 5]).reshape(-1, 1)

        batch_action_values = self.q_network(b_s).gather(1, b_a)
        next_batch_action_values = self.target_q_network(b_s_).max(1)[0].reshape(-1, 1)

        targets = b_r + 0.99 * (1 - b_dones) * next_batch_action_values

        loss = self.loss_fn(batch_action_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_dict = self.q_network.state_dict()
        target_dict = self.target_q_network.state_dict()

        for k in q_dict.keys():
            target_dict[k] = q_dict[k] * 0.005 + target_dict[k] * 0.995

        self.target_q_network.load_state_dict(target_dict)

        return loss.item()


if __name__ == '__main__':
    env = gymnasium.make('CartPole-v1', render_mode="rgb_array")
    agent = Agent()
    memory = Memory()
    for i_episode in trange(600):
        observation, _ = env.reset()
        for t in range(500):
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)

            # Store the transition in memory
            memory.push(observation, action, reward, observation_, terminated)

            # Perform one step of the optimization (on the policy network)
            agent.train(memory)

            # Move to the next state
            observation = observation_

            if terminated or truncated:
                break

    env = gymnasium.make('CartPole-v1', render_mode="human")
    for i_episode in range(100):
        # Initialize the environment and get its state
        observation, _ = env.reset()
        for t in range(200):
            action = agent.action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        print('Episode {} finished after {}', i_episode)
