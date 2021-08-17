from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

from sklearn.feature_selection import SelectFromModel

# 1. data
x, y = load_boston(return_X_y=True)

# print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=66)

# 2. model
model = XGBRegressor(n_jobs=-1)

# 3. train
model.fit(x_train, y_train) # 0.9999977961022865

# 4. pred eval
score = model.score(x_train, y_train) # r2
print(score)

thresholds = np.sort(model.feature_importances_)
# print(thresholds)
# [0.00291435 0.0034828  0.00671642 0.00685145 0.00821344 0.01547304
#  0.01930322 0.03052581 0.03163415 0.05089369 0.07860955 0.16772042
#  0.5776617 ]

print("=======================================")
for thresh in thresholds:
    # print(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape, select_x_test.shape)

    seletion_model = XGBRegressor(n_jobs=-1)
    seletion_model.fit(select_x_train, y_train)

    y_pred = seletion_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" 
            %(thresh, select_x_train.shape[1], score*100))



'''
# 과적합 방지
1. 훈련 lot
2. Dropout
3. normalization , regulation, batchNomal - 정규화 (L1, L2)
4. feature delete 
'''

'''
Thresh=0.003, n=13, R2: 89.41%
Thresh=0.003, n=12, R2: 89.05%
Thresh=0.007, n=11, R2: 90.01%
Thresh=0.007, n=10, R2: 90.06%
Thresh=0.008, n=9, R2: 90.18% ###
Thresh=0.015, n=8, R2: 89.45%
Thresh=0.019, n=7, R2: 88.80%
Thresh=0.031, n=6, R2: 89.67%
Thresh=0.032, n=5, R2: 87.33%
Thresh=0.051, n=4, R2: 85.51%
Thresh=0.079, n=3, R2: 84.80%
Thresh=0.168, n=2, R2: 76.47%
Thresh=0.578, n=1, R2: 46.05%
'''