
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, 
                input_shape=(32,32,3))
vgg16.trainable = False # Freeze weight or train

model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))

model.summary()

'''
Total params: 14,760,789
Trainable params: 46,101
Non-trainable params: 14,714,688
'''

# print(len(model.weights)) # 30
# print(len(model.trainable_weights)) # 30 : T, 4 : F
