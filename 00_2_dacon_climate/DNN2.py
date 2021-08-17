import pandas as pd
import numpy as np
import re
import os
from konlpy.tag import Okt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

import pickle

# etc..
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

def data_loading(path,target_columns):
    train_path = os.path.join(path,'train.csv')
    test_path = os.path.join(path,'test.csv')

    train = pd.read_csv(train_path)
    train_texts = train[target_columns + ['label']]
    test_texts = pd.read_csv(test_path)[target_columns]
    
    print('DATA LOADING DONE')
    
    return train_texts, test_texts
def clean_text(sent):
    sent_clean=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]", " ", sent) #특수문자 및 기타 제거
    sent_clean = re.sub(' +', ' ', sent_clean) # 다중 공백 제거
    return sent_clean

def data_preprocessing(data,target_columns):
    data = data.fillna('NONE')
    data['요약문_연구목표'] = data.apply(lambda x : x['과제명'] if x['요약문_연구목표'] == 'NONE' else x['요약문_연구목표'], axis=1)
    
    okt = Okt()
    data['요약문_한글키워드'] = data.apply(lambda x : ','.join(okt.nouns(x['과제명'])) if x['요약문_한글키워드'] == 'NONE' else x['요약문_한글키워드'], axis = 1)
    
    data.loc[:,target_columns] = data[target_columns].applymap(lambda x : clean_text(x))
    
    return data

def drop_short_texts(train, target_columns):
    train_index = set(train.index)
    for column in target_columns:
        train_index -= set(train[train[column].str.len() < 10].index)

    train = train.loc[list(train_index)]
    
    print('SHORT TEXTS DROPED')
    
    return train

def sampling_data(train, target_columns):
    pj_name_len = 18
    summ_goal_len = 210
    summ_key_len = 18

    max_lens = [pj_name_len, summ_goal_len, summ_key_len]
    total_index = set(train.index)
    for column, max_len in zip(target_columns, max_lens) : 
        temp = train[column].apply(lambda x : len(x.split()) < max_len)
        explained_ratio = temp.values.sum() / train.shape[0]
        
        total_index -= set(train[temp == False].index)
    train = train.loc[list(total_index)].reset_index(drop = True)
    
    return train

def oversampling_minor_classes(train, target_columns) : 
    temp = train.copy()
    temp.loc[:,target_columns] = temp.loc[:,target_columns].applymap(lambda x : len(x.split()))
    pj_range = (6,11)
    summ_goal_range = (30,89)
    summ_key_range = (5,9)
    temp = temp.query('label != 0')
    temp = pd.DataFrame(list(train.loc[temp.query('과제명 in @pj_range or 요약문_연구목표 in @summ_goal_range or 요약문_한글키워드 in @summ_key_range').index].values)*1, columns = temp.columns)
    train = pd.concat([train, temp], axis = 0).reset_index(drop = True)
    train.groupby(['label'])['과제명'].agg('count').plot.bar()
    train.groupby(['label'])['과제명'].agg('count')[1:].plot.bar()
    
    return train

def get_topk_nums(train, target_columns):
    top_ks = []
    for column in target_columns:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train[column])


        threshold = 5 #기준치
        total_cnt = len(tokenizer.word_index) # 단어의 수
        rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

        # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value

            # 단어의 등장 빈도수가 threshold보다 작으면
            if(value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value
       
        top_ks.append(total_cnt - rare_cnt)
    return top_ks

def ngram_vectorize(train_data, label, test_data, top_k) : 
    kwargs = {
    
        'ngram_range' : (1,2),
        'dtype' : 'int32',
        'strip_accents' : False,
        'lowercase' : False,
        'decode_error' : 'replace',
        'analyzer': 'char',
        'min_df' : 2,
        
            }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)

    selector = SelectKBest(f_classif, k=min(80000,top_k))
    selector.fit(x_train, label.values)
    x_train = selector.transform(x_train).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    
    return x_train, x_test

def vectorize_data(train, test, top_ks, target_columns):
    train_inputs = []
    test_inputs = []
    for top_k, column in zip(top_ks, target_columns):
        train_input, test_input = ngram_vectorize(train[column], train['label'], test[column], min(80000,top_k))
        train_inputs.append(train_input)
        test_inputs.append(test_input)
        
    return train_inputs, test_inputs

def data_loading_and_setting_main():
    
    # kwargs
    path = './00_2_dacon_climate/_data'

    
    target_columns = ['과제명','요약문_연구목표','요약문_한글키워드']

    ########################################################################
    train_texts, test_texts = data_loading(path, target_columns)
    
    ########################################################################
    train_texts = data_preprocessing(train_texts,target_columns)
    print('TRAIN NA DATA NUM : ', train_texts.isna().sum().sum())

    test_texts = data_preprocessing(test_texts,target_columns)
    print('TEST NA DATA NUM : ', train_texts.isna().sum().sum())
    
    ########################################################################
    train_texts = drop_short_texts(train_texts, target_columns)
    
    ########################################################################
    
    ########################################################################
    train_texts = sampling_data(train_texts, target_columns)
    
    ########################################################################
    train_texts = oversampling_minor_classes(train_texts, target_columns)

    ########################################################################
    top_ks = get_topk_nums(train_texts, target_columns)
    
    ########################################################################
    train_inputs, test_inputs = vectorize_data(train_texts, test_texts, top_ks, target_columns)
    
    return train_inputs, test_inputs, train_texts['label']

train_inputs, test_inputs, labels = data_loading_and_setting_main()

with open('inputs.pkl', 'wb') as f:
    pickle.dump((train_inputs, test_inputs, labels), f)

import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from glob import glob

def get_input_dataset(data, index, train = False) : 
    input0 = tf.convert_to_tensor(data[0][index].toarray(), tf.float32)
    input1 = tf.convert_to_tensor(data[1][index].toarray(), tf.float32)
    input2 = tf.convert_to_tensor(data[2][index].toarray(), tf.float32)
    
    if train : 
        label = labels[index]

        return input0, input1, input2, label
    else:
        return input0, input1, input2,

def single_dense(x, units):
    fc = Dense(units, activation = None, kernel_initializer = 'he_normal')(x)
    batch = BatchNormalization()(fc)
    relu = ReLU()(batch)
    dr = Dropout(0.2)(relu)
    
    return dr

def create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate):
    x_in0 = Input(input_shape0,)
    x_in1 = Input(input_shape1,)
    x_in2 = Input(input_shape2,)
    
    fc0 = single_dense(x_in0, 512)
    fc0 = single_dense(fc0, 256)
    fc0 = single_dense(fc0, 128)
    fc0 = single_dense(fc0, 64)
    
    fc1 = single_dense(x_in1, 1024)
    fc1 = single_dense(fc1, 512)
    fc1 = single_dense(fc1, 256)
    fc1 = single_dense(fc1, 128)
    fc1 = single_dense(fc1, 64)
    
    fc2 = single_dense(x_in2, 512)
    fc2 = single_dense(fc2, 256)
    fc2 = single_dense(fc2, 128)
    fc2 = single_dense(fc2, 64)
    
    fc = Concatenate()([fc0,fc1,fc2])
    
    fc = single_dense(fc, 128)
    fc = single_dense(fc, 64)
    
    x_out = Dense(num_labels, activation = 'softmax')(fc)
    
    model = Model([x_in0,x_in1,x_in2], x_out)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    return model

with open('inputs.pkl','rb') as f :
    train_inputs, test_inputs, labels = pickle.load(f)

num_labels = 46
learning_rate = 5e-2
seed = np.random.randint(2**16-1)
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)

for train_idx, valid_idx in skf.split(train_inputs[0], labels):
    X_train_input0, X_train_input1, X_train_input2, X_train_label = get_input_dataset(train_inputs, train_idx, train = True)
    X_valid_input0, X_valid_input1, X_valid_input2, X_valid_label = get_input_dataset(train_inputs, valid_idx, train = True)
    
    now = datetime.now()
    now = str(now)[11:16].replace(':','h')+'m'
    ckpt_path = f'./{now}.ckpt'
    
    input_shape0 = X_train_input0.shape[1]
    input_shape1 = X_train_input1.shape[1]
    input_shape2 = X_train_input2.shape[1]


    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor = 'val_acc', save_best_only= True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.9, patience = 2,),
                ]
    model = create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate)
    model.fit(
                        [X_train_input0,X_train_input1,X_train_input2],
                        X_train_label,
                        epochs=1000,
                        callbacks=callbacks,
                        validation_data=([X_valid_input0, X_valid_input1, X_valid_input2], X_valid_label),
                        verbose=1,  # Logs once per epoch.
                        batch_size=4096)
    
    model.load_weights(ckpt_path)
    prediction = model.predict([test_inputs[0], test_inputs[1], test_inputs[2]])
    np.save(f'{now}_prediction.npy', prediction)

predictions = []
for ar in glob('*.npy'):
    arr = np.load(ar)
    predictions.append(arr)

sample = pd.read_csv('../data/sample_submission.csv')
sample['label'] = np.argmax(np.mean(predictions,axis=0), axis = 1)
sample.to_csv('./Day0815.csv',index=False)


with open('inputs.pkl','rb') as f :
    train_inputs, test_inputs, labels = pickle.load(f)

