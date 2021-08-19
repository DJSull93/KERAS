# Practice
# make cancer smote
# f1 score
# label 0 -> 112 delete 

from xgboost import  XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings=warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score

# 1-1. data
datasets = load_breast_cancer()

# 1. data
x = datasets.data
y = datasets.target

# print(pd.Series(y).value_counts())

# print(x.shape, y.shape) # (569, 30) (569,)
y = np.array(y).reshape(569,1)
# print(y.shape)

dasd = np.concatenate((x,y), axis=1)
# print(dasd)

dasd = dasd[dasd[:, 30].argsort()]
# print(dasd)

x = dasd[112:,0:-1] 
y = dasd[112:,-1]

# print(x.shape, y.shape)
# print(y)

# print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66) # , stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# print(pd.Series(y_train).value_counts())

# 2. model
model = XGBClassifier(n_jobs=-1)

# 3. train
model.fit(x_train, y_train, eval_metric='mlogloss')

# 4. eval
score = model.score(x_test, y_test)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred)

print("==================SMOTE==================")

st = time.time()
smote = SMOTE(random_state=66, k_neighbors=60)
et = time.time() - st
x_smote, y_smote = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote).value_counts())

# 2. model
model2 = XGBClassifier(n_jobs=-1)

# 3. train
model2.fit(x_smote, y_smote, eval_metric='mlogloss')

# 4. eval
score2 = model2.score(x_test, y_test)

y_pred2 = model2.predict(x_test)
f12 = f1_score(y_test, y_pred2)

print("before smote :", x_train.shape, y_train.shape)
print("after smote  :", x_smote.shape, y_smote.shape)
print("before somote labels :\n",pd.Series(y_train).value_counts())
print("after somote labels  :\n",pd.Series(y_smote).value_counts())

print("model_best_score_default :", score)
print("model_best_score_smote   :", score2)
print("f1_score_default:", f1)
print("f1_score_smote  :", f12)

'''
before smote : (365, 30) (365,)
after smote  : (580, 30) (580,)
before somote labels :
 1.0    290
0.0     75
dtype: int64
after somote labels  :
 0.0    290
1.0    290
dtype: int64
model_best_score_default : 0.9456521739130435
model_best_score_smote   : 0.9565217391304348
f1_score_default: 0.9640287769784173
f1_score_smote  : 0.9710144927536231
'''