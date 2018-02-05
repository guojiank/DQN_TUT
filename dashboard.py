from collections import deque

import gym
import numpy as np
import tensorflow as tf


class BrainDQN:

    def __init__(self):
        self.replayMemory = deque()
        self.createQNetwork()

    def createQNetwork(self):
        pass

    def trainQNetwork(self):
        pass

    def getAction(self, observation):
        pass


    def add_layer(inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


def playGame():
    env = gym.make('CartPole-v0')

    brain = BrainDQN()

    for episode in range(20):
        observation = env.reset()

        for t in range(100):
            env.render()
            action = brain.getAction(observation)
            observation_, reward, done, _ = env.step(action)
            observation = observation_
            if done:
                print("游戏得:{} ".format(t+1))
                break


