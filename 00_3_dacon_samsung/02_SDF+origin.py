from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
pd.set_option('display.max_columns', 30)

import tensorflow as tf

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
scaler = MaxAbsScaler()
scaler.fit(x_data) 
x_data = scaler.transform(x_data) 
x_eval = scaler.transform(x_eval) 

x_train2 = np.load('./00_3_dacon_samsung/_data/train_xorigin.npy')
x_eval2 = np.load('./00_3_dacon_samsung/_data/test_xorigin.npy')

xtr_all = np.concatenate((x_data, x_train2), axis=1)
xev_all = np.concatenate((x_eval, x_eval2), axis=1)

# print(xtr_all.shape) # (30341, 2064)
# print(y_data.shape) # (30341,)
# print(xev_all.shape) # (602, 2064)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(2200, input_dim=2064, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mae', optimizer='adam',
      metrics=['mse'])
    return model

#validation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import time 
start_time = time.time()
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', 
    KerasRegressor(build_fn=create_deep_learning_model, epochs=250)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=4)
results = cross_val_score(pipeline, xtr_all, y_data, cv=kfold)
print("%.4f (%.4f) MAE" % (results.mean(), results.std()))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
        restore_best_weights=True, verbose=1)

model = create_deep_learning_model()
hist = model.fit(xtr_all, y_data, 
        batch_size=512,  validation_split=0.02, callbacks=[es],
          epochs = 1500)

test_y = model.predict(xev_all)

ss['ST1_GAP(eV)'] = test_y

path2 = './00_3_dacon_samsung/_save/'

ss.to_csv(path2+"orSDF_mlp.csv",index=False)
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print("final score = ", loss[-10])
print("final val score = ", val_loss[-10])
end_time = time.time() - start_time
print("total taim =", end_time)
