import math
from collections import deque
import random

import gymnasium
import numpy as np
import torch.optim
from torch import nn, FloatTensor, BoolTensor, LongTensor
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

    def push(self, state, action, reward, next_state):
        if next_state is not None:
            next_state = FloatTensor(next_state)
        self.q.append((FloatTensor(state), LongTensor([action]), FloatTensor([reward]), next_state))

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
        self.optimizer = torch.optim.AdamW(self.q_network.parameters())
        self.counter = 0

    def choose_action(self, state):
        e = 0.05 + (0.9 - 0.05) * math.exp(-1. * self.counter / 1000)
        self.counter += 1
        if random.random() < e:
            return random.choice([0, 1])
        else:
            return self.q_network(FloatTensor(np.array([state]))).argmax().item()

    def train(self, m: Memory):
        if m.size() < 128:
            return
        batch = list(zip(*m.sample(128)))

        b_s = torch.cat(batch[0]).reshape(-1, 4)
        b_a = torch.cat(batch[1]).reshape(-1, 1)
        b_r = torch.cat(batch[2]).reshape(-1, 1)

        not_none_next_s = torch.cat([i for i in batch[3] if i is not None]).reshape(-1, 4)
        next_s_mask = torch.BoolTensor([[i is not None] for i in batch[3]])

        batch_action_values = self.q_network(b_s).gather(1, b_a)

        next_batch_action_values = torch.zeros(128).reshape(-1, 1)

        next_batch_action_values[next_s_mask] = self.target_q_network(not_none_next_s).max(1).values

        targets = b_r + 0.99 * next_batch_action_values

        loss = self.loss_fn(batch_action_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        q_dict = self.q_network.state_dict()
        target_dict = self.target_q_network.state_dict()

        for k in q_dict.keys():
            target_dict[k] = q_dict[k] * 0.005 + target_dict[k] * 0.995

        self.target_q_network.load_state_dict(target_dict)


if __name__ == '__main__':
    env = gymnasium.make('CartPole-v1', render_mode="human")
    agent = Agent()
    memory = Memory()
    scores = deque([], maxlen=10)
    with trange(600) as bar:
        for i_episode in bar:
            observation, _ = env.reset()
            for t in range(500):
                action = agent.choose_action(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)

                # Store the transition in memory
                memory.push(observation, action, reward, None if terminated else observation_)

                # Perform one step of the optimization (on the policy network)
                agent.train(memory)

                # Move to the next state
                observation = observation_

                if terminated or truncated:
                    scores.append(t)
                    score = sum(scores) / len(scores)
                    bar.set_description(f'score: {score}')
                    break
