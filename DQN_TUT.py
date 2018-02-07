import tensorflow as tf

import numpy as np

import gym

'''
使用dqn玩 openai中的CartPole-v0游戏
'''

tf_r = tf.placeholder(dtype=tf.float32, shape=[None])
tf_s = tf.placeholder(dtype=tf.float32, shape=[None, 4])
tf_s_ = tf.placeholder(dtype=tf.float32, shape=[None, 4])

with tf.variable_scope("Q_function"):
    layer1 = tf.layers.dense(inputs=tf_s, units=30, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(layer1, rate=0.1)
    layer2 = tf.layers.dense(inputs=dropout1, units=30, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(layer2, rate=0.1)
    # inputs tensor shape : [None,4]
    # outputs tensor shape : [None,2]
    Q = tf.layers.dense(inputs=dropout2, units=2)

with tf.variable_scope("Q_target"):
    target_layer1 = tf.layers.dense(inputs=tf_s_, units=30, activation=tf.nn.relu, trainable=False)
    target_dropout1 = tf.layers.dropout(target_layer1, rate=0.1)
    target_layer2 = tf.layers.dense(inputs=target_dropout1, activation=tf.nn.relu, trainable=False)
    dropout2 = tf.layers.dropout(target_layer2, rate=0.1)
    # inputs tensor shape : [None,4]
    # outputs tensor shape : [None,2]
    Q_ = tf.layers.dense(inputs=dropout2, units=2, trainable=False)

target_Q = tf_r + 0.9 * tf.reduce_max(Q_, 1)
tf_loss = target_Q - Q

env = gym.make("CartPole-v0")

for episode in range(1000):
    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation_, award, done, info = env.step(action)
        env.render()
        observation = observation_
        if done:
            break
