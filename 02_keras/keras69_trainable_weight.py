import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

# print(model.weights)
'''
[<tf.Variable 'dense/###kernel###:0' shape=(1, 3) dtype=float32, 
    -> kernel = weight // shape=(input, output)

    ### numpy=array([[-0.63604885, -0.79527956,  1.1079813 ]], ### 
    -> first random weight

    dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) 
dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
numpy=array([[ 1.0456331 , -0.6224754 ],
            [ 0.72483623, -1.0728266 ],
            [-0.7517996 , -0.71320534]], 
       dtype=float32)>, 
<tf.Variable 'dense_1/bias:0' shape=(2,) 
dtype=float32, numpy=array([0., 0.], dtype=float32)>, 

<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, 
numpy=array([[-1.0538802],
             [ 0.3444537]], 
       dtype=float32)>, 
<tf.Variable 'dense_2/bias:0' shape=(1,)
dtype=float32, numpy=array([0.], dtype=float32)>]
'''

# print(model.trainable_weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, 
numpy=array([[ 0.675418 , -0.35249072, -0.8145864 ]], 
dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, 
numpy=array([0., 0., 0.], dtype=float32)>, 

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
numpy=array([[-0.00300193, -0.7420335 ],
            [ 0.79954493, -0.04709399],
            [-1.093619  , -0.5044076 ]], dtype=float32)>, 
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, 
numpy=array([0., 0.], dtype=float32)>, 

<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, 
numpy=array([[ 0.20073092],
             [-0.03805995]], dtype=float32)>, 
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, 
numpy=array([0.], dtype=float32)>]
'''

print(len(model.weights)) # 6 
print(len(model.trainable_weights)) # 6
