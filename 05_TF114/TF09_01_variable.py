import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(77)

W = tf.compat.v1.Variable(tf.random_normal([1]), name='Weight')
# W = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
print(W)

# 1. Session type 1
sess = tf.compat.v1.Session() # open session
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa :", aaa) # aaa : [1.014144]
sess.close() # close session

# 2. Session type 2
sess = tf.InteractiveSession() # open session
sess.run(tf.global_variables_initializer())
bbb = W.eval() # variable.eval()
print("bbb :", bbb) # bbb : [1.014144]
sess.close() # close session

# 3. Session type 3
sess = tf.Session() # open session
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc :", ccc) # ccc : [1.014144]
sess.close() # close session
