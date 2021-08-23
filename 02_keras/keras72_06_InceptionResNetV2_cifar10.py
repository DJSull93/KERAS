# Practice
# cifar 10, 100
# trainable F, T / GlobalAVGP, Flatten / val acc, loss

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2
from tensorflow.keras.applications import ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.python.keras.layers.core import Dropout

# 1. data cifa10
x_train = np.load('./_save/_NPY/k55_x_data_cifar10_train.npy')
x_test = np.load('./_save/_NPY/k55_x_data_cifar10_test.npy')
y_train = np.load('./_save/_NPY/k55_y_data_cifar10_train.npy')
y_test = np.load('./_save/_NPY/k55_y_data_cifar10_test.npy')

# 1. data cifa100
# x_train = np.load('./_save/_NPY/k55_x_data_cifar100_train.npy')
# x_test = np.load('./_save/_NPY/k55_x_data_cifar100_test.npy')
# y_train = np.load('./_save/_NPY/k55_y_data_cifar100_train.npy')
# y_test = np.load('./_save/_NPY/k55_y_data_cifar100_test.npy')

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
m = InceptionResNetV2(weights='imagenet', include_top=False, 
                input_shape=(32*3,32*3,3))
m.trainable = True # Freeze weight or train
# m.trainable = False # Freeze weight or train

model = Sequential()

model.add(UpSampling2D(size=(3,3)))
model.add(m)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax')) # cifar10
# model.add(Dense(100, activation='softmax')) # cifar100

# 3. comple fit // metrics 'acc'
from tensorflow.keras.optimizers import Adam

op = Adam(lr = 0.001)

model.compile(loss='categorical_crossentropy', 
                optimizer=op, metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, 
                mode='min', verbose=1, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.8)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=20, batch_size=256, verbose=2,
    validation_split=0.05, callbacks=[es, lr])
end_time = time.time() - start_time

# 4. predict eval 

loss = model.evaluate(x_test, y_test, batch_size=128)
print("======================================")

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("total time : ", np.round(end_time/60, 0), 'min')
print('acc : ',np.round(acc[-10], 5))
print('val_acc : ',np.round(val_acc[-10], 5))
print('loss : ',np.round(loss[-10], 5))
print('val_loss : ',np.round(val_loss[-10], 5))

'''
###############my acc###############
cifar 10
acc :  0.75103
val_acc :  0.64280
loss :  0.72036
val_loss :  1.07393

cifar 100
acc :  0.55283
val_acc :  0.43813
loss :  1.60040
val_loss :  2.25513

###############cifar10###############
trainable F / GlobalAVGP


trainable F / Flatten

trainable T / GlobalAVGP

trainable T / Flatten


###############cifar100###############
trainable F / GlobalAVGP

trainable F / Flatten

trainable T / GlobalAVGP

trainable T / Flatten

'''