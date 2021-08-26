import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1], [1,2,3,2],
          [3,1,3,4], [4,1,5,5],
          [1,7,5,5], [1,2,5,6],
          [1,6,6,6], [1,7,6,7]]    # (8,4)
y_data = [[0,0,1], [0,0,1],
          [0,0,1], [0,1,0],
          [0,1,0], [0,1,0],
          [1,0,0], [1,0,0]]  # (8,3), one hot encoded

x = tf.placeholder(tf.float32, shape=(None,4)) 
y = tf.placeholder(tf.float32, shape=(None,3))

W = tf.Variable(tf.random.normal([4,3]), name='weight')
b = tf.Variable(tf.random.normal([1,3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 

# optimizer = tf.train.AdamOptimizer(learning_rate=0.0000011)
# train = optimizer.minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(6001):
        cost_val, _ = sess.run([cost, optimizer],
            feed_dict={x:x_data, y:y_data})
        if epochs % 1000 == 0:
            print(epochs, "cost :", cost_val)

    predict = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print(predict, sess.run(tf.argmax(predict, 1)))

'''
5800 cost : 0.40031236 
5900 cost : 0.3984927 
6000 cost : 0.39669994 
[[9.7030985e-01 2.9603237e-02 8.6998247e-05]] [0]
'''