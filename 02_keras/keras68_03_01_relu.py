# Relu // +

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1) # 100

def relu(x):
    return np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# // =. elu / selu / leacky relu