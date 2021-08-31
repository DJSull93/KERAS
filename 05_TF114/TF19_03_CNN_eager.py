# upper 0.996 

from keras.datasets import mnist
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
datasets = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = datasets.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 784)/255.
x_test = x_test.reshape(x_test.shape[0], 784)/255.

learning_rate = 0.0006
epochs = 23
batch_size = 256
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, shape = (None, 784))
y = tf.compat.v1.placeholder(tf.float32, shape = (None, 10))

# 2. model
w1 = tf.compat.v1.get_variable('weight1', shape = [784, 256], initializer = tf.initializers.he_normal()) 
print("w1 : ",w1) 
b1 = tf.Variable(tf.random.normal([1,256], name = 'bias1'))    
print("b1 : ",b1) 
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1) 
print("layer1 : ", layer1) 


w2 = tf.compat.v1.get_variable('weight2', shape = [256, 128], initializer = tf.initializers.he_normal())
b2 = tf.Variable(tf.random.normal([1,128], name = 'bias2'))    
layer2 = tf.nn.elu(tf.matmul(layer1, w2) + b2) 

w3 = tf.compat.v1.get_variable('weight3', shape = [128, 32], initializer = tf.initializers.he_normal())
b3 = tf.Variable(tf.random.normal([1,32], name = 'bias3'))   
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3) 

w4 = tf.compat.v1.get_variable('weight4', shape = [32, 10])
b4 = tf.Variable(tf.random.normal([1,10], name = 'bias4'))   
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

# 3. compile train

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1)) 

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
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

    prediction = tf.compat.v1.equal(tf.compat.v1.arg_max(hypothesis,1), tf.compat.v1.arg_max(y,1))
    accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(prediction, dtype=tf.float32))
    print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# Acc : 0.9789