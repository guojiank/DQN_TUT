import tensorflow as tf
from tensorflow import keras
import random
import numpy as np


def f(x):
    return x * 2 + 4


x_data = [random.randint(0, 100000) for i in range(0, 10000)]

y_data = [(f(x) + random.randrange(-100, 100) * random.random()) for x in x_data]

x_ = np.array(x_data).reshape([len(x_data), 1])
y_ = np.array(y_data).reshape([len(y_data), 1])

x_test = [random.randint(0, 100000) for i in range(0, 10000)]
y_test = [f(x) for x in x_test]

x_t = np.array(x_test).reshape([len(x_test), 1])
y_t = np.array(y_test).reshape([len(y_test), 1])

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.relu))
model.add(keras.layers.Dense(2, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, ))

model.compile(optimizer=tf.train.AdadeltaOptimizer(0.01), loss=tf.losses.mean_squared_error, metrics=['mae'])
model.fit(x=x_, y=y_, batch_size=100, epochs=100)
model.evaluate(x=x_t, y=y_t)


