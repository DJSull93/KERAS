import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
x_train = to_categorical(x_train)
x_test = to_categorical(x_test)

x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255

# 2. model

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['acc'], 
                    loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [10,20,30,40,50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batchsize" : batches, "optimizer" : optimizer,
            "dropout" : dropout}


hyperparameter = create_hyperparameter()
# model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model2 = KerasClassifier(build_fn=build_model, verbose=1, 
                        ) # validation_split=0.2) # epochs=2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = GridSearchCV(model2, hyperparameter, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3,
            validation_split=0.2)


print('best_params_ :',model.best_params_)
print('best_estimator_ :',model.best_estimator_)
print('best_score_ :',model.best_score_)
acc = model.score(x_train, y_train)
print('final_score :', acc)

'''
best_params_ : {'batch_size': 1000, 'drop': 0.3, 'optimizer': 'adam'}     
best_estimator_ : <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000028196E13CD0>
best_score_ : 0.9383499920368195
final_score : 0.9659666419029236
'''