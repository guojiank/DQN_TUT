import tensorflow as tf

import numpy as np

import gym

'''
使用dqn玩 openai中的CartPole-v0游戏
'''

MEMORY_COUNTER = 0
MEMORY_CAPACITY = 2000
MEMORY = np.zeros((MEMORY_CAPACITY, 4 * 2 + 2))
LEARNING_STEP_COUNTER = 0
Learn_step = 0

tf_r = tf.placeholder(dtype=tf.float32, shape=[None, ])
tf_s = tf.placeholder(dtype=tf.float32, shape=[None, 4])
tf_a = tf.placeholder(dtype=tf.int32, shape=[None, ])
tf_s_ = tf.placeholder(dtype=tf.float32, shape=[None, 4])

with tf.variable_scope("e"):
    layer1 = tf.layers.dense(inputs=tf_s, units=30, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(layer1, rate=0.1)
    layer2 = tf.layers.dense(inputs=dropout1, units=30, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(layer2, rate=0.1)
    # inputs tensor shape : [None,4]
    # outputs tensor shape : [None,2]
    Q = tf.layers.dense(inputs=dropout2, units=2)

with tf.variable_scope("t"):
    target_layer1 = tf.layers.dense(inputs=tf_s_, units=30, activation=tf.nn.relu, trainable=False)
    target_dropout1 = tf.layers.dropout(target_layer1, rate=0.1)
    target_layer2 = tf.layers.dense(inputs=target_dropout1, units=30, activation=tf.nn.relu, trainable=False)
    dropout2 = tf.layers.dropout(target_layer2, rate=0.1)
    # inputs tensor shape : [None,4]
    # outputs tensor shape : [None,2]
    Q_ = tf.layers.dense(inputs=dropout2, units=2, trainable=False)
# [None,1] +0.9[None,1]
target_Q = tf_r + 0.9 * tf.reduce_max(Q_, 1)

a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)
q_r_a = tf.gather_nd(params=Q, indices=a_indices)

loss = tf.reduce_mean(tf.squared_difference(target_Q, q_r_a))
train = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def choose_action(s):
    s = s[np.newaxis, :]
    if np.random.uniform() < 0.9:
        action_values = sess.run(Q, feed_dict={tf_s: s})
        action = np.argmax(action_values)
    else:
        action = np.random.randint(0, 2)
    return action


def store_transition(s, a, r, s_):
    global MEMORY_COUNTER
    trainsition = np.hstack((s, [a, r], s_))
    index = MEMORY_COUNTER % MEMORY_CAPACITY
    MEMORY[index, :] = trainsition
    MEMORY_COUNTER += 1


def learn():
    global LEARNING_STEP_COUNTER
    global Learn_step
    if LEARNING_STEP_COUNTER % 150 == 0:
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="e")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="t")
        sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        Learn_step += 1
        print(Learn_step)
    LEARNING_STEP_COUNTER += 1
    sample_index = np.random.choice(MEMORY_CAPACITY, 32)
    b_memory = MEMORY[sample_index, :]
    b_s = b_memory[:, :4]
    b_a = b_memory[:, 4].astype(int)
    b_r = b_memory[:, 5]
    b_s_ = b_memory[:, -4:]
    sess.run(train, feed_dict={tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_})


env = gym.make("CartPole-v0")
env = env.unwrapped
for episode in range(1000):
    observation = env.reset()
    time_life = 0
    while True:

        env.render()
        action = choose_action(observation)
        observation_, award, done, info = env.step(action)

        # r = -1 if done else 1

        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        store_transition(observation, action, r, observation_)

        if MEMORY_COUNTER > MEMORY_CAPACITY:
            learn()

        if done:
            print("over {}".format(time_life))
            break

        observation = observation_
        time_life += 1
