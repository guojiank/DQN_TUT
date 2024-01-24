import random
import time
import collections
import gymnasium
import numpy as np
import random


# def predict(table: dict, s: int, env):
#     return np.argmax(table[s])
#
#
def sample_action(table: dict, s: int, env):
    if random.random() >= 0.9:
        print('random')
        return env.action_space.sample()
    else:
        mx = np.max(table[s])
        arr = np.argwhere(table[s] == mx)
        print(arr)
        a = random.choice(arr)
        print(a)
        return a[0]


def update_table(table: dict, s: int, action: int, r: float, s_: int):
    table[s][action] = r + 0.9 * (max(table[s_]))
    print('update', s, action, r, table[s][action])


def train(table: dict):
    print('start train')
    env = gymnasium.make('FrozenLake-v1', render_mode="human", is_slippery=False)
    for episode in range(200):
        print(f'----------- episode: {episode} -----------')
        observation, _ = env.reset()

        for i in range(50):
            env.render()
            action = sample_action(table, observation, env)
            observation_, reward, terminated, truncated, info = env.step(action)

            update_table(table, observation, action,
                         -1 if (terminated and reward == 0) or (observation == observation_) else reward, observation_)
            observation = observation_
            if terminated:
                print('episode finished after {} time steps'.format(i))
                break


# def test(table: dict):
#     print('start test')
#     env = gym.make('FrozenLake-v1', render_mode="human")
#     observation, _ = env.reset()
#
#     for i in range(20):
#         env.render()
#         action = sample_action(table, observation, env)
#         observation_, reward, terminated, truncated, info = env.step(action)
#         print('observation, action', observation, action)
#
#         update_table(table, observation, action, reward, observation_)
#         observation = observation_
#         if terminated:
#             print('episode finished after {} time steps'.format(i))
#             break

def callback(*args):
    print('hello', args)


if __name__ == '__main__':
    q_table = collections.defaultdict(lambda: np.zeros(4))
    train(q_table)
    # test(q_table)
    # env = gymnasium.make('FrozenLake-v1', render_mode="rgb_array")
    # env.reset()
    # PlayableGame(env, keys_to_action={})
