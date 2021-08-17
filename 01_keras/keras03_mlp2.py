from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
x1 = np.transpose(x)
print(x1.shape) #( 10,3)
y = np.array([[11,12,13,14,15,16,17,18,19,20]])
y1 = np.transpose(y)
print(y.shape) # (10, ) <-> (10, 1)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1, y1, epochs=100, batch_size=1)

# 4. 평가 예측
x_pred = np.array([[10, 1.3, 1]])
loss = model.evaluate(x1, y1)
print('loss : ', loss)
result = model.predict(x_pred)
print('[10, 1.3, 1]의 예측값 : ', result)

y_predict = model.predict(x1)
plt.scatter(x1[:,:1],y1)
plt.plot(x1,y_predict, color='red')
plt.show()

"""
epo = 1200
loss :  0.00026537469238974154
[10, 1.3, 1]의 예측값 :  [[20.004646]]
epo = 600
loss :  0.00019260201952420175
[10, 1.3, 1]의 예측값 :  [[19.996614]]
"""