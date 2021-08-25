# Practice
# TF09_1 copy
# 3 way to print hyperthesis

import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(77)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

# 1. Session type 1
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print("first hypothesis : ", aaa)
sess.close()
# first hypothesis :  [1.3       1.6       1.9000001]

# 2. Session type 2
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval() 
print("second hypothesis : ",bbb)
sess.close()
# second hypothesis :  [1.3       1.6       1.9000001]

# 3. Session type 3
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("third hypothesis : ", ccc)
sess.close()
# third hypothesis :  [1.3       1.6       1.9000001]

