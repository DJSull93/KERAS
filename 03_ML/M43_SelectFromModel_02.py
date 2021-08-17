# Practice
# 1. 상단 모델에 그리드/랜덤 서치로 튜님한 모델 구성
#  최적의 r2값과 피쳐임포턴스 구할것

# 2. 위 스레드값으로 Model Selction 돌려서 최적의 피처 갯수 구할 것

# 3. 위 피쳐 갯수로 피쳐 갯수 조정한뒤 다시 랜덤/그리드 서치
#  최적의 r2값 구할 것

# 4. 1 / 3 번 값 비교  # 0.47

from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sklearn.feature_selection import SelectFromModel

# 1. data
datasets = load_diabetes()

x_data = datasets.data
y_data = datasets.target

datadf = pd.DataFrame(x_data, columns=datasets.feature_names)
# ['age#', 'sex', 'bmi', 'bp', 's1#', 's2', 's3', 's4', 's5', 's6#']

x_data = datadf[['sex', 'bmi', 'bp', 's2', 's3', 's4', 's5']]

# print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
      test_size=0.12, shuffle=True, random_state=1234)

# 2. model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split

# parameters = [{
#     "n_estimators": [100, 200], # =. epochs, default = 100
#     "max_depth": [6, 8, 10, 12],
#     "min_samples_leaf": [3, 5, 7, 10],
#     "min_samples_split": [2, 3, 5, 10],
#     "n_jobs": [-1] # =. qauntity of cpu; -1 = all
# }]

# model = GridSearchCV(XGBRegressor(), parameters)
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, min_samples_leaf=3, min_samples_split=5,
             monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',  
             validate_parameters=1, verbosity=None)

# 3. train
model.fit(x_train, y_train) 

# 4. pred eval
score = model.score(x_test, y_test) # r2
print('r2_score :', score)

# print('Best estimator : ', model.best_estimator_)
# print('Best score  :', model.best_score_)

# import matplotlib.pyplot as plt

# def plot_feature_importance_dataset(model):
#       n_features = datasets.data.shape[1]
#       plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#       plt.yticks(np.arange(n_features), datasets.feature_names)
#       plt.xlabel("Feature Importances")
#       plt.ylabel("Features")
#       plt.ylim(-1, n_features)

# plot_feature_importance_dataset(model)
# plt.show()

# thresholds = np.sort(model.feature_importances_)

# print(thresholds)
# # [0.00291435 0.0034828  0.00671642 0.00685145 0.00821344 0.01547304
# #  0.01930322 0.03052581 0.03163415 0.05089369 0.07860955 0.16772042
# #  0.5776617 ]

# print("=======================================")
# for thresh in thresholds:
#     # print(thresh)
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     # print(select_x_train.shape, select_x_test.shape)

#     seletion_model = XGBRegressor(n_jobs=-1)
#     seletion_model.fit(select_x_train, y_train)

#     y_pred = seletion_model.predict(select_x_test)

#     score = r2_score(y_test, y_pred)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" 
#             %(thresh, select_x_train.shape[1], score*100))


'''
# 1
GridSearchCV
Best estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, min_samples_leaf=3, min_samples_split=5,
             missing=nan, monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',  
             validate_parameters=1, verbosity=None)
Best score  : 0.33842787019825427

RandomizedSearchCV
Best estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, min_samples_leaf=3, min_samples_split=5,
             missing=nan, monotone_constraints='()', n_estimators=100,
             n_jobs=-1, num_parallel_tree=1, random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',  
             validate_parameters=1, verbosity=None)
Best score  : 0.33842787019825427
'''

'''
# 2 
Thresh=0.026, n=10, R2: 41.41%
Thresh=0.048, n=9, R2: 44.33%
Thresh=0.050, n=8, R2: 44.54%
Thresh=0.065, n=7, R2: 51.58% ### 
Thresh=0.068, n=6, R2: 43.18%
Thresh=0.075, n=5, R2: 43.36%
Thresh=0.081, n=4, R2: 47.97%
Thresh=0.088, n=3, R2: 35.29%
Thresh=0.173, n=2, R2: 10.71%
Thresh=0.326, n=1, R2: -3.29%
'''

'''
# 3
r2_score : 0.5158071603957941
'''

'''
# 4
1-> find best_params : r2_score > 0.33842787019825427
3-> cut columns      : r2_score > 0.5158071603957941
'''