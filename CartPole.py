from collections import deque
from itertools import count

import gymnasium
from tqdm import trange

from core.sample_agent import Agent

if __name__ == '__main__':
    env = gymnasium.make('CartPole-v1', render_mode="human")
    agent = Agent(input_size=4, output_size=2, batch_size=128)
    scores = deque([], maxlen=10)
    with trange(600) as bar:
        observation, _ = env.reset()
        for i_episode in bar:
            for t in count():
                action = agent.sample(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                # Store the transition in memory
                agent.push(observation, action, reward, observation_, terminated)
                # Perform one step of the optimization (on the policy network)
                agent.learn()
                # Move to the next state
                observation = observation_
                if terminated or truncated:
                    scores.append(t)
                    score = sum(scores) / len(scores)
                    bar.set_description(f'score: {score}')
                    observation, _ = env.reset()
                    break
