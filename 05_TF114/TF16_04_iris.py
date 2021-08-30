# Practice
# acc score

import tensorflow as tf
import numpy as np
tf.set_random_seed(66)
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
datasets = load_iris()

x_data = datasets.data # (150, 4) 
y_data = datasets.target.reshape(-1,1) # (150, 1)

# print(x_data.shape, y_data.shape)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.25,  random_state = 77)

x = tf.placeholder(tf.float32, shape=(None,4)) 
y = tf.placeholder(tf.float32, shape=(None,3))

W = tf.Variable(tf.zeros([4,3]), name='weight')
b = tf.Variable(tf.zeros([1,3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(7001):
        cost_val, _ = sess.run([cost, optimizer],
            feed_dict={x:x_train, y:y_train})
        if epochs % 1000 == 0:
            print(epochs, "cost :", cost_val)

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    y_test = np.argmax(y_test, axis= 1)
    print('acc_score : ', accuracy_score(y_test, y_pred))

'''
[0.00473118 0.42025766 0.5750112 ]] [1 2 2 1 0 2 2 2 0 1 0 2 0 0 0 2 2 2 0 1 0 2 2 
1 2 1 2 0 1 2 1 2 0 2 2 1 12]
acc_score :  0.9736842105263158
'''