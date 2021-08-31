# upper 0.7

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.client.session import Session
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False / 1.14 // False / 2.41
print(tf.__version__)

tf.compat.v1.set_random_seed(66)

# 1. data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

learning_rate = 0.00014
epochs = 220
batch_size = 1024
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 2. model
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 3, 32])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)

w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 32])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')

w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 32, 64])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)

w33 = tf.compat.v1.get_variable('w33', shape=[3, 3, 64, 64])
L33 = tf.nn.conv2d(L3, w33, strides=[1,1,1,1], padding='SAME')
L33 = tf.nn.relu(L33)
L3_maxpool = tf.nn.max_pool(L33, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')

# w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 64, 128])
# L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='SAME')
# L4 = tf.nn.relu(L4)

# w44 = tf.compat.v1.get_variable('w44', shape=[3, 3, 128, 128])
# L44 = tf.nn.conv2d(L4, w44, strides=[1,1,1,1], padding='SAME')
# L44 = tf.nn.relu(L44)
# L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
#                             strides=[1,2,2,1], # 2, 2 -> pooling size
#                             padding='SAME')
# print(L4_maxpool)

# Flatten
L_flat = tf.compat.v1.reshape(L3_maxpool, (-1, 8*8*64))

w5 = tf.compat.v1.get_variable('w5', shape=[8*8*64, 128])
b5 = tf.Variable(tf.random.normal([128]), name='b5')
L5 = tf.compat.v1.matmul(L_flat, w5) + b5
L5 = tf.nn.relu(L5)

w6 = tf.compat.v1.get_variable('w6', shape=[128, 84])
b6 = tf.Variable(tf.random.normal([84]), name='b6')
L6 = tf.compat.v1.matmul(L5, w6) + b6
L6 = tf.nn.relu(L6)

w7 = tf.compat.v1.get_variable('w7', shape=[84, 10])
b7 = tf.Variable(tf.random.normal([10]), name='b6')
L7 = tf.compat.v1.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)

# 3. compile train

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1)) 

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    avg_loss = 0

    for epochs in range(epochs):
        
        for i in range(total_batch): # 50,000 / 100 = 500
            start = i * batch_size
            end = start + batch_size
            batch_x, batch_y = x_train[start:end], y_train[start:end]
            feed_dict = {x:batch_x, y:batch_y}

            batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

            avg_loss = batch_loss/total_batch
    
        print('Epochs : ', '%04d' %(epochs+1), 'loss : {:.9f}'.format(avg_loss))

    prediction = tf.compat.v1.equal(tf.compat.v1.arg_max(hypothesis,1), tf.compat.v1.arg_max(y,1))
    accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(prediction, dtype=tf.float32))
    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# Epochs :  0100 loss : 0.019202127
# Acc : 0.5279

# Epochs :  0090 loss : 0.012972048
# Acc : 0.6776