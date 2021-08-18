import numpy as np
aaa = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])

# outlier 기준 : 통상적으로 4분위수를 기준 지표로 활용
# 기준 예 // 데이터 평균의 2배 까지를 정상 범주로 판단 
# -> 15223/11 * 2 = ~2767

# 4분위수 : 1사분위 중위수 3사분위 : 2, 6, 90 (정렬 후 분위) 
# Quartile
# IQR : Interquartile Range = 3rd - 1st = 88
# 1~3 분위 : 50% , 각 3분위 너머, 1분위 미만에 1.5배 만큼 
# 범위에 대해 1~3 분위값 만큼 덧셈 뺄셈 해주어 기준 확립 

# outlier function 
def outlier(data_out):
    q0, quartile_1, q2, quartile_3, q4 = np.percentile(data_out, [0, 25, 50, 75, 100])
    print("Q1 : ", quartile_1)
    print("Q2 : ", q2)
    print("Q3 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    # 정상 데이터 범위 지정
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outlier(aaa)

print("outlier at :", outliers_loc) 
# Q0 :  -10000.0
# Q1 :  3.0
# Q2 :  6.0
# Q3 :  49.0
# Q4 :  5000.0
# outlier at : (array([ 2, 10], dtype=int64),)

'''
이상치 처리 
1. 삭제
2. Nan 처리후 보간 // linear
3. -------------- 결측치 처리 방법과 유사
4. scaler -> Rubster, Quantile scaler : 이상치에 둔감함
5. modeling -> tree 계열, XG, LGBM, DT, RF
'''

# # box ploting 
# import matplotlib.pyplot as plt

# plt.boxplot(aaa)
# plt.show()

'''
IQR이란, Interquartile range의 약자로써 Q3 - Q1 구간 의미

백분위수(Percentile. 百分位數)는 크기가 있는 값들로 이뤄진 자료를 순서대로 
나열했을 때 백분율로 나타낸 특정 위치의 값을 이르는 용어

Quartile은 평균과 관련된 개념의 값이 아니고 전체 데이터 set에서 일정 
포지션에 위치 하고 있는 값
'''