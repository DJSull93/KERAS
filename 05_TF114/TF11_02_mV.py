import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[73,51,65],
            [92,98,11],
            [89,31,33],
            [99,33,100],
            [17,66,79]] # (5,3)
y_data = [[152],[185],[180],[205],[142]]    # (5,1)


x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * W + b # 행렬 연산 에러 발생 
hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(y-hypothesis))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

sess.close()