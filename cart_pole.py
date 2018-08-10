import gym
import time
import tensorflow as tf
from tensorflow import keras



model = keras.Sequential()
model.add(keras.layers.Dense(4))

env = gym.make("CartPole-v0")
while True:
    observation = env.reset()
    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print(observation, reward, done, info)
        print(env.action_space)
        print(env.action_space.sample())

        time.sleep(1)
        if done:
            break
