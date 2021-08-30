from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.client.session import Session
from keras.models import Sequential
from keras.layers import Conv2D

tf.set_random_seed(66)

# 1. data
datasets = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = datasets.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.00001
epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 2. model
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
# print(L1) # shape=(?, 28, 28, 32)
L1 = tf.nn.relu(L1)
# print(L1) # shape=(?, 28, 28, 32)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')

# print(L1_maxpool) # shape=(?, 14, 14, 32)

w2 = tf.get_variable('w2', shape=[3, 3, 32, 64], 
        initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')
# print(L2) # shape=(?, 14, 14, 64)
# print(L2_maxpool) # shape=(?, 7, 7, 64)

w3 = tf.get_variable('w3', shape=[3, 3, 64, 128], 
        initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')
# print(L3) # shape=(?, 7, 7, 128)
# print(L3_maxpool) # shape=(?, 4, 4, 128)

w4 = tf.get_variable('w4', shape=[2, 2, 128, 64], 
        initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')
# print(L4) # shape=(?, 4, 4, 64)
# print(L4_maxpool) # shape=(?, 2, 2, 64)

# Flatten
L_flat = tf.reshape(L4_maxpool, (-1, 2*2*64))
# print(L_flat) # (?, 256)

w5 = tf.get_variable('w5', shape=[2*2*64, 64])
b5 = tf.Variable(tf.random.normal([64]), name='b5')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
# print(L5) # shape=(?, 64)

w6 = tf.get_variable('w6', shape=[64, 32])
b6 = tf.Variable(tf.random.normal([32]), name='b6')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
# print(L6) # shape=(?, 32)

w7 = tf.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random.normal([10]), name='b6')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
# print(hypothesis) # shape=(?, 10)

# 3. compile train

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    avg_loss = 0

    for epochs in range(epochs):
        
        for i in range(total_batch): # 60,000 / 100 = 600
            start = i * batch_size
            end = start + batch_size
            batch_x, batch_y = x_train[start:end], y_train[start:end]
            feed_dict = {x:batch_x, y:batch_y}

            batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

            avg_loss = batch_loss/total_batch
    
        print('Epochs : ', '%04d' %(epochs+1), 'loss : {:.9f}'.format(avg_loss))

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    y_test = np.argmax(y_test, axis= 1)
    print('acc_score : ', accuracy_score(y_test, y_pred))

# Epochs :  0015 loss : 0.004384255
# acc_score :  0.2392