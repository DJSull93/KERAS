################# 앞뒤가 똑같은 오~토인코더 #################

import numpy as np
from tensorflow.keras.datasets import mnist

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype("float32")/255.
x_test = x_test.reshape(10000, 28,28,1).astype("float32")/255.

x_train_noise = x_train + np.random.normal(0, 0.2, size=x_train.shape)
x_test_noise = x_test + np.random.normal(0, 0.2, size=x_test.shape)

x_train_noise = np.clip(x_train_noise, a_min=0, a_max=1)
x_test_noise = np.clip(x_test_noise, a_min=0, a_max=1)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D

def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(28, 28, 1),
                activation='relu', padding='same'))
    model.add(MaxPooling2D(1,1))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
    
    model.add(UpSampling2D(size=(1,1)))

    model.add(Conv2D(1, (2, 2), activation='sigmoid', padding='same'))
    return model

model = autoEncoder(hidden_layer_size=154)

# 3. compile, train
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noise, x_train, epochs=10, batch_size=512)

# 4. eval pred
output = model.predict(x_test_noise)

# # 5. visualize
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# original image
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# noised image
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# original image
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noise[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('noise', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()