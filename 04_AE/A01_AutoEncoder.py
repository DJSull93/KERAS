# Encode / Decode -> 암호화, 복호화
# 이미지 데이터 활용에 필요, -> 발전 : GAN
################# 앞뒤가 똑같은 오~토인코더 #################
# input =. output
# x -> x, don't need y label

# GAN 명확하게 나오지만 AE는 뿌옇게 나옴 

import numpy as np
from tensorflow.keras.datasets import mnist

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32")/255.
x_test = x_test.reshape(10000, 784).astype("float32")/255.

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout

ip_img = Input(shape=(784,))

# case 1
# encode = Dense(64, activation='relu')(ip_img)
# decode = Dense(784, activation='relu')(encode)
# decode = Dense(784, activation='linear')(encode)
# decode = Dense(784, activation='softmax')(encode)
# decode = Dense(784, activation='tanh')(encode)
# decode = Dense(784, activation='sigmoid')(encode)
# 분류모델이 아니라 다양한 활성함수가 가능하지만, 제한 범위만큼 결과값 다름

# case 2
encode = Dense(1064, activation='relu')(ip_img)
decode = Dense(784, activation='softmax')(encode)


autoEncoder = Model(ip_img, decode) 

# autoEncoder.summary()
'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________

##### 인풋과 아웃풋은 동일하나, 중간 64를 통과하며 작은 특성은 도태
'''

# 3. compile, train
# autoEncoder.compile(optimizer='adam', loss='mse')
autoEncoder.compile(optimizer='adam', loss='binary_crossentropy')

autoEncoder.fit(x_train, x_train, epochs=30,
                batch_size=256 ,validation_split=0.2)

# 4. pred eval
decode_img = autoEncoder.predict(x_test)

# # 5. visualize
import matplotlib.pyplot as plt

n= 10
for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decode_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()