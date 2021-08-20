# VGG16 : Visual Geometry Group 16 layer

# Imagenet : 1,000개의 클래스로 구성되며 총 백만 개가 넘는 데이터
#  약 120만 개는 학습(training)에 쓰고, 5만개는 검증(validation)
#  학습 데이터셋 용량은 약 138GB, 검증 데이터셋 용량은 약 6GB
#  학습 데이터 각 클래스당 약 1,000개가량의 사진으로 구성



import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19

# model = VGG16(weights='imagenet', include_top=False, 
                # input_shape=(32,32,3))

model = VGG19()

model.summary()


