# Practice
# Single layer -> MLP

import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

# 1. data
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. model
x = tf.placeholder(tf.float32, shape=[None,2]) 
y = tf.placeholder(tf.float32, shape=[None,1])

# 2-1. hidden layer
W0 = tf.Variable(tf.random.normal([2,11]), name='weight')
b0 = tf.Variable(tf.random.normal([11]), name='bias')

layer0 = tf.sigmoid(tf.matmul(x, W0) + b0)

# 2-2. hidden layer2
W1 = tf.Variable(tf.random.normal([11,10]), name='weight')
b1 = tf.Variable(tf.random.normal([10]), name='bias')

layer1 = tf.sigmoid(tf.matmul(layer0, W1) + b1)

# 2-3. output layer
W2 = tf.Variable(tf.random.normal([10,1]), name='weight')
b2 = tf.Variable(tf.random.normal([1]), name='bias')

layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# binary_crossentropy 
cost = -tf.reduce_mean(y*tf.log(layer2)+(1-y)*tf.log(1-layer2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

predicted = tf.cast(layer2 > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost, layer2, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 1000 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

h, c, a = sess.run([layer2, predicted, accuracy], feed_dict = {x:x_data,y:y_data})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)

sess.close()

'''
Hypothesis :
 [[0.00571369]
 [0.99078393]
 [0.99461764]
 [0.0077499 ]]
predict :
 [[0.]
 [1.]
 [1.]
 [0.]] 
 Accuarcy :  1.0
'''