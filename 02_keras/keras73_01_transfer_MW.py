# 가장 잘나온 전이학습으로 
# 학습 결과치 도출
# 59번 or 61 과 성능 비교

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.25
)

test_datagen = ImageDataGenerator(rescale=1./255)

x_train = np.load('./_save/_NPY/k59_mw_x_train.npy')
x_test = np.load('./_save/_NPY/k59_mw_x_test.npy')
y_train = np.load('./_save/_NPY/k59_mw_y_train.npy')
y_test = np.load('./_save/_NPY/k59_mw_y_test.npy')
x_pred = np.load('./_save/_NPY/k59_mw_x_pred.npy')

# print(xy_train[0][0].shape) # (8005, 150, 150, 3)
# print(xy_train[0][1].shape) # (8005, 2)

augment_size = 400

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

x_argmented = x_argmented.reshape(x_argmented.shape[0], 150, 150, 3) # (32, 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3) # (160, 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3) # (120, 150, 150, 3)

x_argmented = train_datagen.flow(x_argmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_argmented)) 
y_train = np.concatenate((y_train, y_argmented)) 

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
model.add(Dense(1, activation= 'sigmoid'))

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=2,
    validation_split=0.1, callbacks=[es]
               )
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print("total time : ", np.round(end_time/60, 0), 'min')
print('acc : ',np.round(acc[-10], 5))
print('val_acc : ',np.round(val_acc[-10], 5))
# print('loss : ',np.round(loss[-10], 5))
print('val_loss : ',np.round(val_loss[-10], 5))        

y_predict = model.predict([x_pred])
res = (1-y_predict) * 100
print('남자일 확률 : ',res, '%')

'''
without flow
acc :  0.938035249710083
val_acc :  0.5513078570365906

with InceptionV3 Trainable False, Flatten
total time :  1.0 min
acc :  0.83803
val_acc :  0.56055
val_loss :  0.68781
남자일 확률 :  [[0.82218647]] %
'''