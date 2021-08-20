# VGG16 : Visual Geometry Group 16 layer

# Imagenet : 1,000개의 클래스로 구성되며 총 백만 개가 넘는 데이터
#  약 120만 개는 학습(training)에 쓰고, 5만개는 검증(validation)
#  학습 데이터셋 용량은 약 138GB, 검증 데이터셋 용량은 약 6GB
#  학습 데이터 각 클래스당 약 1,000개가량의 사진으로 구성

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16, VGG19

model = VGG16(weights='imagenet', include_top=False, 
                input_shape=(100,100,3))
model.trainable = False

# model = VGG16()

# print(len(model.weights)) # 26
# print(len(model.trainable_weights)) # 26 : T, 0 : F

model.summary()

# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# ...
# flatten (Flatten)            (None, 25088)             0
# fc1 (Dense)                  (None, 4096)              102764544
# fc2 (Dense)                  (None, 4096)              16781312
# predictions (Dense)          (None, 1000)              4097000

# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# fc = fully connected / 이전 레이어의 출력을 "평탄화"하여 
# 다음 스테이지의 입력이 될 수 있는 단일 벡터로 변환
