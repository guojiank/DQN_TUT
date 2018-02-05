import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

c = tf.stack([a, b], axis=0)
d = tf.stack([a, b], axis=1)

e = tf.unstack(d, axis=0)
f = tf.unstack(d, axis=1)

with tf.Session() as sess:
    print('原始向量数据：')
    print(sess.run(a))
    print(sess.run(b))
    print('矩阵拼接的函数示例,得到一个矩阵：')  # 返回值是多维矩阵
    print('以"0维"的方式进行拼接')
    print(sess.run(c))
    print('以"1维"的方式进行拼接')
    print(sess.run(d))

    print('矩阵分解的函数示例，得到一个list：')  # 返回值是一个list
    print('以"0维"的方式进行分解')
    print(sess.run(e))
    print('以"1维"的方式进行分解')
    print(sess.run(f))