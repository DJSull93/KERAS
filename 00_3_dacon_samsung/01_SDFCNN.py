from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', 30)

# 1-1. data
datasets = load_boston()

# 1. data

path = './00_3_dacon_samsung/_data/'

data1 = pd.read_csv(path+"train_sdf.csv")
data2 = pd.read_csv(path+"dev_sdf.csv")
data3 = pd.read_csv(path+"test_sdf.csv")
ss = pd.read_csv(path+"sample_submission.csv")

datasets = pd.concat([data1, data2], axis=0)

datasets = datasets.drop(columns=['Unnamed: 0','uid','SMILES','S1_energy(eV)', 'T1_energy(eV)'])

# print(datasets)
# abonds atoms bonds dbonds HBA1 HBA2 HBD logP MP MR MW nF 
# rotors sbonds tbonds TPSA     y

x_data = datasets.iloc[:,:-1]
y_data = datasets.iloc[:,-1]

x_eval = data3.drop(columns=['Unnamed: 0','uid','SMILES'])
x_eval = x_eval.to_numpy()
# print(x_eval.shape) # (602, 16)
# print(x_data.shape, y_data.shape) # (30341, 16) (30341,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_data) 
x_data = scaler.transform(x_data) 
x_eval = scaler.transform(x_eval) 

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# 3. 컴파일 훈련
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import time 
start_time = time.time()
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', 
    KerasRegressor(build_fn=create_deep_learning_model, epochs=30)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, x_data, y_data, cv=kfold)
print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

model = create_deep_learning_model()
hist = model.fit(x_data, y_data, 
        batch_size=1024,  
          epochs = 100)

# 4. 평가 예측
test_y = model.predict(x_eval)

ss['ST1_GAP(eV)'] = test_y

path2 = './00_3_dacon_samsung/_save/'

ss.to_csv(path2+"SDF.csv",index=False)
loss = hist.history['loss']

print("final score : ", loss[-1])
end_time = time.time() - start_time
print("total taim :", end_time)


