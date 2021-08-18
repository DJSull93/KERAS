# 지도 / 비지도 학습 (y label 유, 무)
# 지도 학습 : SVM, Linear etc..
# 비지도 학습 : Clustering - KMean // y 값을 찾아낼 수 있음

# KMean - 구분 라벨만큼 임의의 점 을 찍고, 값들과의 거리 계산, 
#         가까운 값들 사이로 점 이동, 점과의 거리를 기반으로 
#         최근접 이웃 방식의 y label 생성

# matric - accuracy_score : y, newlabel score 

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

datasets = load_iris()

irisDF = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
# print(irisDF)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
# n_clusters =. number of labels / max_iter =. epochs
kmean.fit(irisDF)

result = kmean.labels_

irisDF['cluster'] = kmean.labels_   # clusting y_label
irisDF['target'] = datasets.target  # original y_label

# print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_result = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
# print(iris_result)


'''
# print(result)
# print(datasets.target)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
 2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
 2 1]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''

'''
# print(iris_result)

0       0          50 /o
1       1          48 /o
        2           2 /x
2       1          14 /x
        2          36 /o
'''