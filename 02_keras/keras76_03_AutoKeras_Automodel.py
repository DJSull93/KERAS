import autokeras as ak

from tensorflow.keras.datasets import mnist

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28*1)
x_test = x_test.reshape(-1, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model

inputs = ak.ImageInput()
outputs = ak.ImageBlock(
    block_type='resnet',
    normalize=True,
    augment=False
)(inputs)
outputs = ak.ClassificationHead()(outputs)

model = ak.AutoModel(
    inputs=inputs, outputs=outputs, overwrite=True, max_trials=1
)
# -> 아래와 동일하나 함수형

# model = ak.ImageClassifier(
#     overwrite=True,
#     max_trials=1,
# )

#3. compile train
import time
st = time.time()
model.fit(x_train, y_train, epochs=5)

et = time.time() - st

#4. eval pred

y_pred = model.predict(x_test)

res = model.evaluate(x_test, y_test)

print(res)
print('time = ', et)

model2 = model.export_model()
model2.summary()
