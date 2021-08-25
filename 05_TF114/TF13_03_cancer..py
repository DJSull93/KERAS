import tensorflow as tf
import numpy as np
tf.set_random_seed(66)
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
datasets = load_breast_cancer()

x_data = datasets.data # (569, 30) 
y_data = datasets.target #  (569,)

# print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

y_data = y_data.reshape(-1,1) # (506, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2,  random_state = 42)

x = tf.placeholder(tf.float32, shape=(None,30)) 
y = tf.placeholder(tf.float32, shape=(None,1))

W = tf.Variable(tf.zeros([30,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, W) + b)
# hypothesis = tf.matmul(x, W) + b

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# cost = tf.reduce_mean(tf.square(y-hypothesis))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00004)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0000011)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32) 

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val,hy_val ,_ = sess.run([cost, hypothesis,train], feed_dict={x:x_train, y:y_train})

        if step % 50 == 0:
            print(f'step : {step} \ncost : {cost_val} \nhy_val : \n{hy_val}')

    h , c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={x:x_test, y:y_test})

    print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')

'''
predict value : [[0.52031285]
 [0.09974188]
 [0.3206721 ]
 [0.677768  ]
 [0.6429453 ]]
"original value:
[[1.]
 [0.]
 [0.]
 [1.]
 [1.]]
accuracy: : 0.9298245906829834
'''