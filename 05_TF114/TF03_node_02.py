# Practice 
# add, sub, div, mul

import tensorflow as tf
from tensorflow.python.client.session import Session

node1 = tf.constant(2.0, tf.float32)

node2 = tf.constant(3.0, tf.float32)

node3 = tf.add(node1, node2)

node4 = tf.subtract(node1, node2)

node5 = tf.div(node1, node2)

node6 = tf.multiply(node1, node2)

sess = Session()

print("node1, node2 : ",sess.run([node1, node2]))
print("add : ", sess.run(node3))
print("sub : ", sess.run(node4))
print("div : ", sess.run(node5))
print("mul : ", sess.run(node6))

'''
node1, node2 :  [2.0, 3.0]
add :  5.0
sub :  -1.0
div :  0.6666667
mul :  6.0
'''
###############################################