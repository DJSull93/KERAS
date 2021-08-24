# y = wx + b

import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(666)

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(1, dtype=tf.float32)
# b = tf.Variable(1, dtype=tf.float32)

# random // normal -> 정규분포
W = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)

hyperthesis = x_train * W + b

loss = tf.compat.v1.reduce_mean(tf.square(hyperthesis - y_train))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

sess = Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2000):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], 
        feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    if step % 20 == 0:
        # print('step :',step, 'loss :', sess.run(loss), 
        #         'W :', sess.run(W), 'b :', sess.run(b))
        print('step :',step, 'loss :', loss_val, 
                'W :', W_val, 'b :', b_val)
