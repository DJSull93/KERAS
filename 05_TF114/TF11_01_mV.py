# Multi Variable

import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.training import optimizer
tf.set_random_seed(7777)

x1_data = [73. , 93., 89. , 96., 73.]       # math
x2_data = [80. , 88., 91. , 98., 66.]       # english
x3_data = [75. , 93., 90. , 100., 70.]      # social
y_data = [152. , 185., 180. , 196., 142.]   # converted score

# x (5,3), y (5, )

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W1 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight2')
W3 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight3')
b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

hyporthesis = x1*W1 + x1*W2 + x1*W3 + b

cost = tf.reduce_mean(tf.square(y-hyporthesis))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hyporthesis, train],
        feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

sess.close()