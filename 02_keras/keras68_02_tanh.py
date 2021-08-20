# Hyperbolic Tangent // -1 ~ 1

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1) # 100

y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()
