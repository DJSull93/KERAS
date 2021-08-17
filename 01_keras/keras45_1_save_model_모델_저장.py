'''
모델과 파라미터가 동일해도 학습마다 결과가 다름

따라서 대회 제출 등 최적의 웨이트를 찾아도 다시금 구현이 어려움

-> 모델 세이브, 모델 체크포인트 활용, 최적의 웨이트 및 모델 저장
-> 불러오기를 통해 재구현 가능

save model : after model -> 단순히 모델만 저장, 웨이트 재학습 필요
save model : after fit -> 학습한 웨이트값 저장, 얼리스탑 무시
Model Check Point : in fit-> 매 에포당 최적의 웨이트 값 갱신 시 저장

모델 저장에는 상단의 세가지가 존재하며 사용법이 다르나, 
사용시 얼리스탑의 결과를 반영하는 Model Check Point이 사용에 적합
'''

# example cifar100

from tensorflow.keras.datasets import cifar100, mnist

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1) 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(28, 28, 1))) 
model.add(Dropout(0.1))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())     

model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                   
model.add(Dropout(0.1))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())       

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(Dropout(0.1))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())       

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

# model.summary()

model.save('./_save/keras45_1_save_model.h5')


# 3. compile fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=2,
    validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

loss = model.evaluate(x_test, y_test, batch_size=128)
print("======================================")
print("total time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

# # 5. plt visualize
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# # plot 1 
# plt.subplot(2,1,1)
# plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
# plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title("loss")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.legend(loc='upper right')

# # plot 2
# plt.subplot(2,1,2)
# plt.plot(hist.history["acc"])
# plt.plot(hist.history["val_acc"])
# plt.grid()
# plt.title("acc")
# plt.ylabel("acc")
# plt.xlabel("epochs")
# plt.legend(['acc', 'val_acc'])

# plt.show()
'''
total time :  81.11641836166382
loss :  0.03077828139066696
acc :  0.9921000003814697

total time :  9.25413990020752
loss :  0.036677632480859756
acc :  0.988099992275238
'''
