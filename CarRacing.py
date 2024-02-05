from itertools import count

import gymnasium as gym
from tqdm import trange

if __name__ == '__main__':
    with gym.make("CarRacing-v2", render_mode="human") as env, trange(300) as bar:
        observation, _ = env.reset()
        for episode in bar:
            for _ in count():
                action = env.action_space.sample()
                observation_, reward, terminated, truncated, _ = env.step(action)
                observation = observation_
                if terminated or truncated:
                    observation, info = env.reset()
                    break
