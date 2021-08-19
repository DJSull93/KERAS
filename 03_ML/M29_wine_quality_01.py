from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

datasets = datasets.values

x = datasets[:,0:11] # (4898, 11)
y = datasets[:,[11]] # (4898, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.15, shuffle=True, random_state=24)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# 2. model
from xgboost import XGBClassifier, XGBRegressor

model = XGBClassifier(n_jobs=-1)

# 3. train
model.fit(x_train, y_train)

# 4. eval pred
score = model.score(x_test, y_test)
print("acc : ", score)

'''
default XGB
acc :  0.6775510204081633
'''