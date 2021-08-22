import numpy as np
from tensorflow.python.keras.layers.core import Dropout

# 1. data
x_train = np.load('./_save/_NPY/k59_cd_x_train.npy')
x_test = np.load('./_save/_NPY/k59_cd_x_test.npy')
y_train = np.load('./_save/_NPY/k59_cd_y_train.npy')
y_test = np.load('./_save/_NPY/k59_cd_y_test.npy')

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

m = InceptionV3(weights='imagenet', include_top=False, 
                input_shape=(150,150,3))
m.trainable = False # Freeze weight or train

model = Sequential()

model.add(m)
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation= 'softmax'))

# 3. compile train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

# hist = model.fit_generator(xy_train, epochs=50,
#  steps_per_epoch=32,
#  validation_data=xy_test,
#  validation_steps=4,
#  callbacks=[es]) # 32 -> 160/5

hist = model.fit(x_train, y_train, epochs=500,
                callbacks=[es],
                validation_split=0.05,
                steps_per_epoch=32,
                validation_steps=1)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# visualize upper data

# print('val_acc : ',val_acc[:-1])

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
print('val_loss : ',val_loss[-10])

'''
without flow
acc :  0.9951341152191162
val_acc :  0.5910224318504333
loss :  0.6094908714294434

with InceptionV3 Trainable False, Flatten
acc :  0.768674373626709
val_acc :  0.7680798172950745
val_loss :  0.4420032501220703
'''