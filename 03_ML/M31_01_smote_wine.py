# 수치 데이터 (분류) - 증폭 // 회귀는 애매함

# 라벨 불균형에 대해 검증 - 재현율, 정밀도, f1 score
# 적은 라벨 증폭 -> smote

# smote -> kneighbors 기반으로 증폭

from imblearn.over_sampling import SMOTE

from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings=warnings.filterwarnings('ignore')

# 1. data
datasets = load_wine()
x = datasets.data # (178, 13) 
y = datasets.target # (178,)

# print(pd.Series(y).value_counts())

# 1    71
# 0    59
# 2    48

# print(y)

# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-30] # (148, 13)
y_new = y[:-30] # (148,) 

# print(x_new.shape, y_new.shape)

# print(pd.Series(y_new).value_counts())

# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new,
      test_size=0.2, shuffle=True, random_state=77, stratify=y_new)
# -> label 비율 맞춰 훈련/테스트 분리

# print(pd.Series(y_train).value_counts())

# 1    53 -> 53
# 0    44 -> 53
# 2    14 -> 53

# 2. model
model = XGBClassifier(n_jobs=-1)

# 3. train
model.fit(x_train, y_train, eval_metric='mlogloss')

# 4. eval
score = model.score(x_test, y_test)


print("==================SMOTE==================")

smote = SMOTE(random_state=77)

x_smote, y_smote = smote.fit_resample(x_train, y_train)

# print(x_smote.shape) # (159, 13) -> 53 * 3 (0, 1, 2)
# print(y_smote.shape) # (159,)

# print(pd.Series(y_smote).value_counts())

# 0    53
# 1    53
# 2    53

# 2. model
model2 = XGBClassifier(n_jobs=-1)

# 3. train
model2.fit(x_smote, y_smote, eval_metric='mlogloss')

# 4. eval
score2 = model2.score(x_test, y_test)

print("before smote :", x_train.shape, y_train.shape)
print("after smote  :", x_smote.shape, y_smote.shape)
print("before somote labels :\n",pd.Series(y_train).value_counts())
print("after somote labels  :\n",pd.Series(y_smote).value_counts())

print("model_best_score_default :", score)
print("model_best_score_smote   :", score2)

'''
before smote : (121, 13) (121,)
after smote  : (180, 13) (180,)
before somote labels :
 1    60
0    50
2    11
dtype: int64
after somote labels  :
 0    60
1    60
2    60
dtype: int64
model_best_score_default : 0.9090909090909091
model_best_score_smote   : 0.9545454545454546
'''

# 결과치 보고 판단해야함