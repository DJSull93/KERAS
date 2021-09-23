import autokeras as ak

from tensorflow.keras.datasets import mnist

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. model
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2,
)

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

'''
ak.ImageRegressor
ak.StructuredDataClassifier
ak.StructuredDataRegressor
'''

'''
[0.040082938969135284, 0.9865999817848206]
time =  77.26785564422607
'''