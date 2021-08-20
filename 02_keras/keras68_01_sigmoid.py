# Sigmoid // 0 ~ 1

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1) # 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# print(len(x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()