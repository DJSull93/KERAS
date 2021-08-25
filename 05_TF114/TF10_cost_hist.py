import tensorflow as tf
from tensorflow.python.client.session import Session
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [2., 4., 6.]

W = tf.placeholder(tf.float32)

hypothesis = x * W

cost = tf.reduce_mean(tf.square(y-hypothesis)) # loss =. cost

W_history = []
cost_history = []

with tf.compat.v1.Session() as sess :
    
    for i in range(-30, 50):
        sess.run(tf.compat.v1.global_variables_initializer())
        curr_W = i
        curr_cost = sess.run(cost, feed_dict={W:curr_W})

        W_history.append(curr_W)
        cost_history.append(curr_cost)

print("****************************************************")
print("W_history :", W_history)
print("****************************************************")
print("cost_history :", cost_history)
print("****************************************************")

plt.plot(W_history, cost_history)
plt.xlabel("W_history")
plt.ylabel("cost_history")
plt.show()