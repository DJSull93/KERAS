# coefficient 

# 1. data
import pandas as pd

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()

df = pd.DataFrame({'X' : x, "Y": y})

x_train = df.loc[:, "X"] # (10, )
y_train = df.loc[:, "Y"] # (10, )
print(x_train.shape, y_train.shape)

x_train = x_train.values.reshape(len(x_train), 1) # (10, ) -> (10, 1)
print(x_train.shape, y_train.shape)

from sklearn.linear_model import LinearRegression

# 2. model
model = LinearRegression()

# 3. train
model.fit(x_train, y_train)

# 4. eval pred
score = model.score(x_train, y_train)
print("score : ", score)

print("weight : ", model.coef_)
print("bias : ", model.intercept_)

'''
score :  1.0
weight :  [2.]
bias :  3.0
'''

