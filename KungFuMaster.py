from itertools import count

import gymnasium as gym
from tqdm import trange

if __name__ == '__main__':
    with gym.make("KungFuMaster-v4", render_mode="human", obs_type='ram') as env, trange(10) as bar:
        observation, _ = env.reset()
        for episode in bar:
            for _ in count():
                action = env.action_space.sample()
                observation_, reward, terminated, truncated, _ = env.step(action)
                print('reward: ', reward)
                observation = observation_
                if terminated or truncated:
                    bar.set_description(f"reward:{reward}")
                    observation, info = env.reset()
                    break
