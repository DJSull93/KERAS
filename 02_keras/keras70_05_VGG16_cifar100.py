# Practice
# VGG16 with cifar100

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32, 32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3) # (10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. model
vgg16 = VGG16(weights='imagenet', include_top=False, 
                input_shape=(32,32,3))
vgg16.trainable = True # Freeze weight or train

model = Sequential()

model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
from tensorflow.keras.optimizers import Adam

op = Adam(lr = 0.001)

# model.compile(loss='categorical_crossentropy', 
#                 optimizer='adam', metrics='acc')
model.compile(loss='categorical_crossentropy', 
                optimizer=op, metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, 
                mode='min', verbose=1, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.8)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=2,
    validation_split=0.05, callbacks=[es, lr])
end_time = time.time() - start_time

# 4. predict eval 

loss = model.evaluate(x_test, y_test, batch_size=256)
print("======================================")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("total time : ", end_time)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])  

'''
with lr reduce
total time :  177.57455277442932
acc :  0.5527999997138977
val_acc :  0.4381333291530609
loss :  1.6004047393798828
val_loss :  2.255129098892212

VGG16 trainable F / GlobalAVGP ###############
total time :  72.63158178329468
acc :  0.5463578701019287
val_acc :  0.3840000033378601
loss :  1.6726396083831787
val_loss :  2.551847457885742

VGG16 trainable T / GlobalAVGP
total time :  184.5901279449463
acc :  0.4687157869338989
val_acc :  0.36160001158714294
loss :  1.8146429061889648
val_loss :  2.5740089416503906

VGG16 trainable F / Flatten
total time :  62.39733266830444
acc :  0.48113682866096497
val_acc :  0.3691999912261963
loss :  1.9557135105133057
val_loss :  2.5301716327667236

VGG16 trainable T / Flatten
total time :  193.59314513206482
acc :  0.48507368564605713
val_acc :  0.34439998865127563
loss :  1.7561811208724976
val_loss :  2.6008431911468506
'''
