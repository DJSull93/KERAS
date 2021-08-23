# 2 번 카피
# CNN deep 구성
# 2 개의 모델 / 베이식, 딥 
# 2 모델 성능 비교
'''
conv2D
Maxpool
Conv2D
Maxpool
Conv2D -> encoder

Conv2D
Upsample
Maxpool
Conv2D
Maxpool
Conv2D
Maxpool
Conv2D(1, ) -> decoder
'''

import numpy as np
from tensorflow.keras.datasets import mnist

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype("float32")/255.
x_train2 = x_train.reshape(60000, 28*28).astype("float32")/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32")/255.

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, LeakyReLU, Conv2DTranspose

def autoEncoder(hidden_layer_size):
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_1 = x

    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_2 = x

    x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x

    x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same', activation='sigmoid')(x+x_1)
    x = Dropout(0.4)(x)
    x = x
    outputs = x
    model = Model(inputs = inputs,outputs=outputs)

    return model

model1 = autoEncoder(hidden_layer_size=154)

# model1.summary()

# 3. compile, train
model1.compile(optimizer='adam', loss='mse')

model1.fit(x_train, x_train, epochs=10, batch_size=512)

# 4. eval pred
output1 = model1.predict(x_test)

# 5. visualize
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10) ) = \
    plt.subplots(2, 5, figsize = (20, 7))


# 이미지 다섯 개 무작위
random_images = random.sample(range(output1.shape[0]), 5)

# 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# basic 오토인코더가 출력한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('basic', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
