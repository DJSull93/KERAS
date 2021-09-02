# multi gpu use

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 1. data
# y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

######################################################################
# multi gpu use
######################################################################
import tensorflow as tf

# strategy = tf.distribute.MirroredStrategy() # -> Error : [[Adam/NcclAllReduce]] [Op:__inference_train_function_3028]
strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
    tf.distribute.HierarchicalCopyAllReduce() # 84
# )
# strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
    # tf.distribute.ReductionToOneDevice() # 84
# )
# strategy = tf.distribute.MirroredStrategy(
    # devices=['/gpu:0'] # 1번 1개만 돌아감
    # devices=['/gpu:1'] # 2번 1개만 돌아감
#     devices=['/gpu:0', '/gpu:1'] # -> 오류남
# )

with strategy.scope():

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                            activation='relu' ,input_shape=(28, 28))) 
    model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
    model.add(MaxPool1D())                                         
    model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
    model.add(Conv1D(64, 2, padding='same', activation='relu'))    
    model.add(Flatten())                                              
    model.add(Dense(256, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

######################################################################

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=576, verbose=2,
    validation_split=0.2)
    # , callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
print('val_loss : ',val_loss[-10])