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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data,y:y_data})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)

sess.close()

'''
5600 cost : 0.03462649 
 [[0.00108409]
 [0.05283239]
 [0.06858116]
 [0.92767626]
 [0.99488515]
 [0.99885225]]
5800 cost : 0.033528093 
 [[0.00100236]
 [0.051327  ]
 [0.06636581]
 [0.92968416]
 [0.9951851 ]
 [0.9989335 ]]
6000 cost : 0.032498002 
 [[9.2903036e-04]
 [4.9905047e-02]
 [6.4290702e-02]
 [9.3158281e-01]
 [9.9545926e-01]
 [9.9900705e-01]]
Hypothesis :
 [[9.2868257e-04]
 [4.9898133e-02]
 [6.4280666e-02]
 [9.3159205e-01]
 [9.9546057e-01]
 [9.9900740e-01]] 
 predict :
 [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]]
 Accuarcy :  1.0
'''