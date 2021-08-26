# Practice 
# r2_score 

import tensorflow as tf
import numpy as np
tf.set_random_seed(77)
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
datasets = load_boston()

x_data = datasets.data # (506, 13) 
y_data = datasets.target # (506, )

# print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

y_data = y_data.reshape(-1,1) # (506, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2,  random_state = 77)

x = tf.placeholder(tf.float32, shape=(None,13)) 
y = tf.placeholder(tf.float32, shape=(None,1))

W = tf.Variable(tf.random.normal([13,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(y-hypothesis))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
optimizer = tf.train.AdamOptimizer(learning_rate=8e-1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val)

predicted = sess.run(hypothesis, feed_dict={x:x_test})

r2 = r2_score(y_test, predicted)
print("R2 : ",r2)

sess.close()

'''
4800 cost : 25.298487
5000 cost : 24.414785
R2 :  0.7292493773895393
'''