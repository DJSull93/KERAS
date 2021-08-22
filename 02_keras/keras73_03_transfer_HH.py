import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)

x_train = np.load('./_save/_NPY/k59_hh_x_train.npy')
x_test = np.load('./_save/_NPY/k59_hh_x_test.npy')
y_train = np.load('./_save/_NPY/k59_hh_y_train.npy')
y_test = np.load('./_save/_NPY/k59_hh_y_test.npy')

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, verbose=2,
    validation_split=0.2, callbacks=[es], steps_per_epoch=32,
                validation_steps=4)
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])                             

'''
without flow
acc :  0.9826839566230774
val_acc :  0.7948718070983887

with InceptionV3 Trainable False, Flatten
acc :  0.9701257944107056
val_acc :  0.7354838848114014
val_loss :  0.5616358518600464
'''