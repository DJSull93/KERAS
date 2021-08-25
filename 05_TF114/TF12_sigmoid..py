import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] # (6,2)
y_data = [[0],[0],[0],[1],[1],[1]] # (6,1)


x = tf.placeholder(tf.float32, shape=[None,2]) 
y = tf.placeholder(tf.float32, shape=[None,1])


W = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
hypothesis = tf.sigmoid(tf.matmul(x, W) + b)

# cost = tf.reduce_mean(tf.square(y-hypothesis))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary_crossentropy 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-2)
train = optimizer.minimize(cost)

# pred = 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

sess.close()

'''
5980 cost : 0.060926005
 [[0.00419192]
 [0.08563906]
 [0.12231328]
 [0.88424426]
 [0.98571163]
 [0.99606127]]
5990 cost : 0.060836375
 [[0.00417746]
 [0.0855374 ]
 [0.1221284 ]
 [0.88437873]
 [0.98574847]
 [0.99607325]]
6000 cost : 0.06074698
 [[0.00416306]
 [0.0854359 ]
 [0.12194386]
 [0.8845127 ]
 [0.9857852 ]
 [0.99608517]]
'''