import gymnasium as gym
from tqdm import trange

if __name__ == '__main__':

    env = gym.make('Humanoid-v4', render_mode='human')

    for episode in trange(1000):
        env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            observation, award, done, _, _ = env.step(action)
            print(award)
            if done:
                break
