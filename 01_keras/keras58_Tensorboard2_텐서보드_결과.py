
'''
텐서보드 결과 확인 프로세스
cmd
d:
cd study
cd _save
cd _graph
dir/w
tensorboard --logdir=.
http://localhost:6006/
'''

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                write_graph=True, write_images=True)

model.fit(x, y, epochs=1000, batch_size=1, callbacks=[tb],
            validation_split=0.2)

# 4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)

y_predict = model.predict(x)
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()
