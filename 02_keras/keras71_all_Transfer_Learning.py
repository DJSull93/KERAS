# Pre-trained model

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2
from tensorflow.keras.applications import ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

# 모델별로 파하미터와 웨이트 수 정리할 것

# m = VGG16()
# Non-trainable params: 138,357,544       
# total weigt num 32
# trainable weigt num 0

# m = VGG19()
# Non-trainable params: 143,667,240
# total weigt num 38
# trainable weigt num 0

# m = Xception()
# Non-trainable params: 22,910,480
# total weigt num 236
# trainable weigt num 0

# m = ResNet50()
# Non-trainable params: 25,636,712
# total weigt num 320
# trainable weigt num 0

# m = ResNet50V2()
# Non-trainable params: 25,613,800
# total weigt num 272
# trainable weigt num 0

# m = ResNet101()
# Non-trainable params: 44,707,176
# total weigt num 626
# trainable weigt num 0

# m = ResNet101V2()
# Non-trainable params: 44,675,560
# total weigt num 544
# trainable weigt num 0

# m = ResNet152()
# Non-trainable params: 60,419,944
# total weigt num 932
# trainable weigt num 0

# m = ResNet152V2()
# Non-trainable params: 60,380,648
# total weigt num 816
# trainable weigt num 0

# m = DenseNet121()
# Non-trainable params: 8,062,504
# total weigt num 606
# trainable weigt num 0

# m = DenseNet169()
# Non-trainable params: 14,307,880
# total weigt num 846
# trainable weigt num 0

# m = DenseNet201()
# Non-trainable params: 20,242,984
# total weigt num 1006
# trainable weigt num 0

# m = InceptionV3()
# Non-trainable params: 23,851,784
# total weigt num 378
# trainable weigt num 0

# m = InceptionResNetV2()
# Non-trainable params: 55,873,736
# total weigt num 898
# trainable weigt num 0

# m = MobileNet()
# Non-trainable params: 4,253,864
# total weigt num 137
# trainable weigt num 0

# m = MobileNetV2()
# Non-trainable params: 3,538,984
# total weigt num 262
# trainable weigt num 0

# m = MobileNetV3Small()
# Non-trainable params: 2,554,968
# total weigt num 210
# trainable weigt num 0

# m = MobileNetV3Large()
# Non-trainable params: 5,507,432
# total weigt num 266
# trainable weigt num 0

# m = NASNetLarge()
# Non-trainable params: 88,949,818
# total weigt num 1546
# trainable weigt num 0

# m = NASNetMobile()
# Non-trainable params: 5,326,716
# total weigt num 1126
# trainable weigt num 0

# m = EfficientNetB0()
# Non-trainable params: 5,330,571
# total weigt num 314
# trainable weigt num 0

# m = EfficientNetB1()
# Non-trainable params: 7,856,239
# total weigt num 442
# trainable weigt num 0

m = EfficientNetB7()
# Non-trainable params: 66,658,687
# total weigt num 1040
# trainable weigt num 0

m.trainable = False

m.summary()
print("total weigt num",len(m.weights)) 
print("trainable weigt num",len(m.trainable_weights))