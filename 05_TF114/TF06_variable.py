import tensorflow as tf
from tensorflow.python.client.session import Session

x = tf.Variable([2], tf.float32, name='test')
y = tf.Variable([3], tf.float32, name='test')
z = tf.Variable([14], tf.float32, name='test')

# Attempting to use uninitialized value test [[{{node _retval_test_0_0}}]]
# 텐서플로1 의 변수는 무조건 초기화 필요함 !!!!!!!!!!!!!!!!!!!!!!!!

init = tf.global_variables_initializer()
# 이전 변수들 모두 초기화

sess = Session()

sess.run(init)

print("x, y, z : ",sess.run([x,y,z])) # x :  [2]