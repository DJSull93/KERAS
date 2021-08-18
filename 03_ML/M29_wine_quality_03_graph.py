# Practice : check outlier
# outlier check with var graph

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

datasets = datasets.values

x = datasets[:,0:11] # (4898, 11)
y = datasets[:,[11]] # (4898, 10)

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)

# bar ploting 
import matplotlib.pyplot as plt

# count_data.plot()
plt.bar(count_data.index, count_data)
plt.show()
# 라벨 단순화 필요