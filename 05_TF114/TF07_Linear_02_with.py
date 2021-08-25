# y = wx + b

import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# sess = Session()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(train)
        if step % 20 == 0:
            print('step :',step, 'loss :', sess.run(loss), 
                    'W :', sess.run(W), 'b :', sess.run(b))

'''
step : 1900 loss : 2.0689856e-06 W : 1.9983293 b : 1.0037978
step : 1920 loss : 1.8794653e-06 W : 1.9984077 b : 1.0036194
step : 1940 loss : 1.7068018e-06 W : 1.9984826 b : 1.0034493
step : 1960 loss : 1.54997e-06 W : 1.9985539 b : 1.0032872
step : 1980 loss : 1.408174e-06 W : 1.9986217 b : 1.0031328
'''