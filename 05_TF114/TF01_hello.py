import tensorflow as tf
print(tf.__version__)

# print("hello world") - doesn't work

Hello = tf.constant("Hello world")
print(Hello) # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(Hello)) # b'Hello world'

