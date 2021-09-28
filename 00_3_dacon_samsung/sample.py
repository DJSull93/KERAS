from openbabel import pybel
from tqdm import tqdm
import numpy as np
import os
import pandas as pds

def sdf_loadtrain(uid,train=True):
    if train:
        paths = os.path.join('.','00_3_dacon_samsung','_data','test_sdf',f'test_{uid}.sdf')
    else:
        paths = os.path.join('.','00_3_dacon_samsung','_data','test_sdf',f'test_{uid}.sdf')
    return paths


train_df = pds.read_csv(os.path.join('.','00_3_dacon_samsung','_data','test.csv'))

mols = dict()

for n in tqdm(train_df.index):
    mol = [i for i in pybel.readfile('sdf',sdf_loadtrain(n))]
    if len(mol) > 0:
        mols[n] = mol[0]

set([i for i in range(train_df.shape[0])]) - set(mols.keys())

mols_df = pds.DataFrame().from_dict({n:v.calcdesc()for n,v in mols.items()}).transpose()
mols_df = mols_df.dropna(axis = 1)
mols_df.loc[:,'uid'] = [f'test_{n}' for n in mols.keys() ]

df = pds.merge(train_df,mols_df,'outer',on='uid').dropna()
# df['y'] = df['S1_energy(eV)'] - df['T1_energy(eV)']

df = df.reset_index(drop=True)
print(df)

df.to_csv('./00_3_dacon_samsung/_data/test_sdf.csv', sep=',', na_rep='NaN')

