from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

import numpy as np
import pandas as pd

ffpp = "pattern"

path = './00_3_dacon_samsung/_data/'

train = pd.read_csv(path+"train.csv")
dev = pd.read_csv(path+"dev.csv")
test = pd.read_csv(path+"test.csv")

ss = pd.read_csv(path+"sample_submission.csv")

train = pd.concat([train,dev])

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']


train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

import math
train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

pd.Series(np_train_fps_array[:,0]).value_counts()

import math
test_fps = [] #test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

# print(np_test_fps_array.shape)
# print(len(test_y))

pd.Series(np_test_fps_array[:,0]).value_counts()

# print(np_test_fps_array.shape)
from tensorflow.keras.optimizers import Adam
op = Adam(learning_rate = 0.0008)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

def create_deep_learning_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',                          
                            activation='relu', input_shape=(2048, 1))) 
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Conv1D(64, 2, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer=op)
    return model

from sklearn.model_selection import train_test_split

X, Y = np_train_fps_array, np.array(train_y)

x_train, x_test, y_train, y_test = train_test_split(X, Y,
      test_size=0.1, shuffle=True, random_state=66)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# print(x_train.shape) # (24276, 2048, 1)
# print(x_test.shape) # (6069, 2048, 1)

#validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=create_deep_learning_model, epochs=25)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=15, 
                mode='min', verbose=1, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.8)

model = create_deep_learning_model()
hist = model.fit(x_train, y_train, 
        batch_size=2048, validation_split=0.1, 
          epochs = 1000, callbacks=[es, lr])

loss = model.evaluate(x_test, y_test, batch_size=2048)

loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-15])
print('val_loss : ',val_loss[-15])

test_y = model.predict(np_test_fps_array)

ss['ST1_GAP(eV)'] = test_y

path2 = './00_3_dacon_samsung/_save/'

ss.to_csv(path2+"cnn_mlp.csv",index=False)
