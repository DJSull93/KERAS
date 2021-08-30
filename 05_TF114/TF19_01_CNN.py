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
# [3, 3, 1, 32] in weight : 
# [3, 3] : kernel_size
# [1] : input channel
# [32] : filters = output channel
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')

print(w1) # shape=(3, 3, 1, 32)
print(L1) # shape=(?, 28, 28, 32)

# Compare with Tensor 2
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
#             padding='same', input_shape=(28, 28, 1)))

'''
##### get_variable #####
w2 = tf.Variable(tf.random.normal([3, 3, 1, 32], name='bias'))
-> Variable : needs initial value // get_variable : don't need
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])
w3 = tf.Variable([1], dtype=tf.float32)

# sess = Session()
# sess.run(tf.global_variables_initializer())
# print(np.min(sess.run(w1)))     # -0.14101818
# print(np.max(sess.run(w1)))     # 0.14186849
# print(np.mean(sess.run(w1)))    # -0.0003214967
# print(np.median(sess.run(w1)))  # 0.0037886351
# print(sess.run(w1))
'''