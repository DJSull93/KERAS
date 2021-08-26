import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape=[None,2]) 
y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
hypothesis = tf.sigmoid(tf.matmul(x, W) + b)

# binary_crossentropy 
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 1000 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data,y:y_data})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)

sess.close()
