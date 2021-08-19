# k_neighbors

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings=warnings.filterwarnings('ignore')

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

datasets = datasets.values

x = datasets[:,0:11] # (4898, 11)
y = datasets[:,11] # (4898,)

# print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=77, stratify=y)

# print(pd.Series(y_train).value_counts())
# 6.0    1758
# 5.0    1166
# 7.0     704
# 8.0     140
# 4.0     130
# 3.0      16
# 9.0       4

# 2. model
model = XGBClassifier(n_jobs=-1)

# 3. train
model.fit(x_train, y_train, eval_metric='mlogloss')

# 4. eval
score = model.score(x_test, y_test)


print("==================SMOTE==================")

smote = SMOTE(random_state=77, k_neighbors=3)
# k_neighbors : int or object, optional (default=5)
# k_neighbors 줄어들면 > 연산 줄어듦 -> 성능 하락

x_smote, y_smote = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote).value_counts())

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
before smote : (3918, 11) (3918,)
after smote  : (12306, 11) (12306,)
before somote labels :
 6.0    1758
5.0    1166
7.0     704
8.0     140
4.0     130
3.0      16
9.0       4
dtype: int64
after somote labels  :
 6.0    1758
5.0    1758
4.0    1758
9.0    1758
8.0    1758
7.0    1758
3.0    1758
dtype: int64
model_best_score_default : 0.6591836734693878
model_best_score_smote   : 0.639795918367347
'''