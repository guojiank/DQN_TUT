import math
from collections import deque
from random import random, sample, choice

import numpy as np
import torch
from torch import FloatTensor, optim, nn, LongTensor, BoolTensor


class Network(nn.Module):
    def __init__(self, input_size, output_size, features):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, output_size)
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:

    def __init__(self, input_size, output_size, memory_capacity=10000, batch_size=64, gamma=0.99, tun=0.005,
                 e_size=1000, features=128):
        self.features = features
        self.e_size = e_size
        self.input_size = input_size
        self.output_size = output_size
        self.memory = Memory(memory_capacity)
        self.policy_net = Network(self.input_size, self.output_size, features).to(device)
        self.target_net = Network(self.input_size, self.output_size, features).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters())
        self.batch_size = batch_size
        self.gamma = gamma
        self.tun = tun
        self.count = 0
        pass

    def push(self, s, a, r, s_, done):
        self.memory.push(
            (FloatTensor(np.array([s])),
             LongTensor(np.array([[a]])),
             FloatTensor(np.array([r])),
             FloatTensor(np.array([s_])),
             BoolTensor(np.array([done])))
        )

    def sample(self, s):
        e = 0.05 + 0.85 * math.exp(-1.0 * self.count / self.e_size)
        self.count += 1
        if random() < e:
            return choice([_ for _ in range(self.output_size)])
        else:
            return self.action(s)

    def action(self, s):
        state = FloatTensor(np.array([s])).to(device)
        return self.policy_net(state).argmax().item()

    def learn(self):
        if self.memory.size() < self.batch_size:
            return
        batch = list(zip(*self.memory.sample(self.batch_size)))
        batch_state = torch.cat(batch[0]).to(device)
        batch_action = torch.cat(batch[1]).to(device)
        batch_reward = torch.cat(batch[2]).to(device)
        batch_next_state = torch.cat(batch[3]).to(device)
        batch_done = torch.cat(batch[4]).to(device)

        next_final_state = batch_next_state[~batch_done]

        batch_action_values = self.policy_net(batch_state).gather(1, batch_action).reshape(-1)

        next_action_values = torch.zeros(self.batch_size).to(device)
        next_action_values[~batch_done] = self.target_net(next_final_state).max(1).values

        batch_target_values = batch_reward + self.gamma * next_action_values

        batch_action_values.to(device)
        batch_target_values.to(device)

        loss = self.loss_fn(batch_action_values, batch_target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        policy_dict = self.policy_net.state_dict()
        target_dict = self.target_net.state_dict()

        for key in policy_dict.keys():
            target_dict[key] = policy_dict[key] * self.tun + target_dict[key] * (1 - self.tun)

        self.target_net.load_state_dict(target_dict)
