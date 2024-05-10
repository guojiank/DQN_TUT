from itertools import count
from time import sleep

import gymnasium as gym
from tqdm import trange

from core.sample_agent import Agent

if __name__ == '__main__':
    with gym.make("LunarLander-v2", render_mode="rgb_array") as env, trange(600) as bar:
        observation, _ = env.reset(seed=42)
        agent = Agent(env.observation_space.shape[0], env.action_space.n, e_size=10000, batch_size=64)
        for episode in bar:
            for _ in count():
                action = agent.sample(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                agent.push(observation, action, reward, observation_, terminated)
                agent.learn()
                observation = observation_
                if terminated or truncated:
                    bar.set_description(f"reward:{reward}")
                    observation, info = env.reset()
                    break

    with gym.make("LunarLander-v2", render_mode="human") as env, trange(10) as bar:
        observation, _ = env.reset(seed=42)
        for episode in bar:
            for _ in count():
                action = agent.action(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                observation = observation_
                if terminated or truncated:
                    bar.set_description(f"reward:{reward}")
                    sleep(2)
                    observation, info = env.reset()
                    break
