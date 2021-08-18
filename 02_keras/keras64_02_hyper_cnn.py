# Practice 
# change in CNN
# epochs 1 or 2
# nodes, activation, epochs[1, 2, 3], lr[0.01, 0.001]

# layer -> variable [Dense, CNN, LSTM etc..]

from sklearn.utils import validation
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28*28,1).astype('float32')/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

def build_model(drop=0.5, node=2, activation='relu', learning_rate=0.001):
    inputs = Input(shape=(28*28,1), name='input')
    x = Conv1D(filters = node, kernel_size=5, activation= activation, 
                padding= 'same', name= 'cnn1')(inputs)
    # x = Dropout(drop) (x)
    # x = Conv1D(filters = node/2, kernel_size=3, activation= activation, 
    #             padding= 'same', name= 'cnn2')(x)
    x = Dropout(drop) (x)       
    # x = Conv1D(filters = node/4, kernel_size=2, activation= activation, 
    #             padding= 'same', name= 'cnn3')(x)
    x = MaxPooling1D(2,2) (x)
    x = Dropout(drop) (x)
    x = Flatten() (x)
    x = Dense(node, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate), metrics=['acc'],
                    loss='categorical_crossentropy')
    return model                    

def create_hyperparameter():
    batches = [100]
    # optimizer = [Adam, RMSprop, Adadelta]
    activation = ['selu','relu']
    dropout = [0.3]
    node = [12, 24]
    epochs = [1, 2]
    validation_split = [0.1]#, 0.2]
    learning_rate = [0.001]#, 0.01]
    return {"batch_size":batches,# "optimizer":optimizer,
            "drop":dropout, "activation":activation,
            "node":node, "epochs":epochs,
            "validation_split":validation_split,
            "learning_rate":learning_rate     
    }

hyperparameter = create_hyperparameter()
# model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',patience=10)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, 
                mode='auto', verbose=1, factor=0.9)

model2 = KerasClassifier(build_fn=build_model, verbose=1, 
                        ) # validation_split=0.2) # epochs=2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = RandomizedSearchCV(model2, hyperparameter, cv=2)

model.fit(x_train, y_train, verbose=1, callbacks = [es, lr])


print('best_params_ :',model.best_params_)
print('best_estimator_ :',model.best_estimator_)
print('best_score_ :',model.best_score_)
acc = model.score(x_train, y_train)
print('final_score :', acc)

'''
best_params_ : {'validation_split': 0.1, 'node': 24, 
            'learning_rate': 0.001, 'epochs': 2, 
            'drop': 0.3, 'batch_size': 100, 'activation': 'relu'}
best_estimator_ : <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000288F46BB640>
best_score_ : 0.9358333349227905
final_score : 0.9466999769210815
'''