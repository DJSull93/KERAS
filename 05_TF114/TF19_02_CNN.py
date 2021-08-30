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

learning_rate = 0.001
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

w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')

# print(L2) # shape=(?, 14, 14, 64)
# print(L2_maxpool) # shape=(?, 7, 7, 64)

w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')

# print(L3) # shape=(?, 7, 7, 128)
# print(L3_maxpool) # shape=(?, 4, 4, 128)

w4 = tf.get_variable('w4', shape=[2, 2, 128, 63])
L4 = tf.nn.conv2d(L3_maxpool, w3, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], # 1, 1 -> 칸채우기용
                            strides=[1,2,2,1], # 2, 2 -> pooling size
                            padding='SAME')

print(L4) # shape=(?, 2, 2, 128)
print(L4_maxpool) # shape=(?, 1, 1, 128)
