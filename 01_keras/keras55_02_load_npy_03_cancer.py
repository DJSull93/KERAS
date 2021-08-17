import numpy as np

x_data = np.load('./_save/_NPY/k55_x_data_cancer.npy')
y_data = np.load('./_save/_NPY/k55_y_data_cancer.npy')

# print(type(x_data), type(y_data)) 
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.25, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                          
                        activation='relu', input_shape=(30, 1))) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation="sigmoid"))

# 3. 컴파일 훈련
# data 형태가 다르므로 mse 대신 binary_crossentropy 사용
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
    validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측


loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
time :  12.296932697296143
loss :  0.12056826800107956
acc :  0.9580419659614563
'''