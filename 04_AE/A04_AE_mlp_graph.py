################# 앞뒤가 똑같은 오~토인코더 #################

import numpy as np
from tensorflow.keras.datasets import mnist

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32")/255.
x_test = x_test.reshape(10000, 784).astype("float32")/255.

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout

def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

outputs = [x_test]

for i in range(6):

    a = 2**i
    print(f'=============node {a} 시작=============')
    model = autoEncoder(hidden_layer_size=a)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    model.fit(x_train, x_train, epochs = 10, batch_size=1024)

    outputs.append(model.predict(x_test))

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize = (15,15))

random_imgs = random.sample(range(outputs[0].shape[0]), 5)

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28),
                cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()