# Softmax // 

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 5) # 100

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()