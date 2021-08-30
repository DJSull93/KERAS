# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from scipy.stats.morestats import Std_dev
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

# 1. data
datasets = tf.keras.datasets.mnist

(x_train, y_train) , (x_test, y_test) = datasets.load_data()

x_train = x_train.reshape(60000, 28*28*1)/255.
x_test = x_test.reshape(10000, 28*28*1)/255.

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
x = tf.placeholder(tf.float32, shape=[None,28*28]) 
y = tf.placeholder(tf.float32, shape=[None,10])

# 2-1. hidden layer
W0 = tf.Variable(tf.random.normal([28*28,270], stddev=0.1, name='weight'))
b0 = tf.Variable(tf.random.normal([1,270], stddev=0.1, name='bias'))
layer0 = tf.nn.relu(tf.matmul(x, W0) + b0)
layer0 = tf.nn.dropout(layer0, keep_prob=0.1)

W1 = tf.Variable(tf.random.normal([270,240], stddev=0.1, name='weight'))
b1 = tf.Variable(tf.random.normal([1,240], stddev=0.1, name='bias'))
layer1 = tf.nn.relu(tf.matmul(layer0, W1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob=0.2)

W14 = tf.Variable(tf.random.normal([240,42], stddev=0.1, name='weight'))
b14 = tf.Variable(tf.random.normal([1,42], stddev=0.1, name='bias'))
layer14 = tf.nn.relu(tf.matmul(layer1, W14) + b14)
# layer14 = tf.nn.dropout(layer14, keep_prob=0.2)

# 2-3. output layer
W2 = tf.Variable(tf.random.normal([42,10], stddev=0.1, name='weight'))
b2 = tf.Variable(tf.random.normal([1,10], stddev=0.1, name='bias'))

layer2 = tf.nn.softmax(tf.matmul(layer14, W2) + b2)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(layer2), axis=1)) 

optimizer = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(131):
        cost_val, _ = sess.run([cost, optimizer],
            feed_dict={x:x_train, y:y_train})
        if epochs % 3 == 0:
            print(epochs, "cost :", cost_val)

    y_pred = sess.run(layer2, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    y_test = np.argmax(y_test, axis= 1)
    print('acc_score : ', accuracy_score(y_test, y_pred))

'''
230 cost : 0.6077514
[7 2 1 ... 4 8 6]
acc_score :  0.8047
'''
