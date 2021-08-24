# ptyhon 3.8.8. env
# -> Session doesn't work

import tensorflow as tf
from tensorflow.python.client.session import Session

tf.compat.v1.disable_eager_execution() # -> to use Session 

print(tf.__version__)

Hello = tf.constant("Hello world")

print(Hello)

# sess = tf.Session() # AttributeError: module 'tensorflow' has no attribute 'Session'

sess = tf.compat.v1.Session() # -> also change version of Session() 

print(sess.run(Hello))





