# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

# 1. data
datasets = tf.keras.datasets.mnist

(x_train, y_train) , (x_test, y_test) = datasets.load_data()

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

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

# 2-1. output layer
W2 = tf.Variable(tf.zeros([28*28,10]), tf.float32, name='weight')
b2 = tf.Variable(tf.zeros([1,10]), tf.float32, name='bias')

layer2 = tf.nn.softmax(tf.matmul(x, W2) + b2)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(layer2), axis=1)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000095).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(401):
        cost_val, _ = sess.run([cost, optimizer],
            feed_dict={x:x_train, y:y_train})
        if epochs % 10 == 0:
            print(epochs, "cost :", cost_val)

    predict = sess.run(layer2, feed_dict={x:x_test})
    print(sess.run(tf.argmax(predict, 1)))

    y_pred = sess.run(layer2, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    y_test = np.argmax(y_test, axis= 1)
    print('acc_score : ', accuracy_score(y_test, y_pred))

'''
[7 2 1 ... 4 5 6]
acc_score :  0.911
'''