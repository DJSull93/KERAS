# Practice : check outlier
# label -> 3,4 :0, 5,6,7 :1, 8,9 :2 change
# 라벨 수정 권한 존재할 시에만 가능함

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

datasets = datasets.values

x = datasets[:,0:11] # (4898, 11)
y = datasets[:,[11]] # (4898, 10)

# 1-1. y data label change
newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]

y = np.array(newlist) # (4898,)

# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
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
acc :  0.9469387755102041
'''