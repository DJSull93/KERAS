# Practice
# make cancer smote
# f1 score

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
before smote : (455, 30) (455,)
after smote  : (568, 30) (568,)
before somote labels :
 1    284
0    171
dtype: int64
after somote labels  :
 0    284
1    284
dtype: int64
model_best_score_default : 0.9736842105263158
model_best_score_smote   : 0.9824561403508771
f1_score_default: 0.979591836734694
f1_score_smote  : 0.9864864864864865
'''