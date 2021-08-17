# coefficient 

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

import matplotlib.pyplot as plt

# plt.plot(x, y)
# plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)