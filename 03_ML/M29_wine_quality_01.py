from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

x = datasets.iloc[:,0:11] # (4898, 11)
y = datasets.iloc[:,[11]] # (4898, 10)