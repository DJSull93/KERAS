import numpy as np

aaa = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])

aaa = aaa.transpose()
print(aaa.shape)

# def outlier(data_out):
#     lis = []
#     for i in range(data_out.shape[1]):
#         quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
#         print("Q1 : ", quartile_1)
#         print("Q2 : ", q2)
#         print("Q3 : ", quartile_3)
#         iqr = quartile_3 - quartile_1
#         lower_bound = quartile_1 - (iqr * 1.5)
#         upper_bound = quartile_3 + (iqr * 1.5)
#         # 정상 데이터 범위 지정
#         print('lower_bound: ', lower_bound)
#         print('upper_bound: ', upper_bound)

#         m = np.where((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
#         lis.append(m)

#     return np.array(lis)

def outliers(data_out):
    allout = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        print('1사분위(25%지점): ',  quartile_1)
        print('q2(50%지점): ',  q2)
        print('3사분위(75%지점): ',  quartile_3)
        iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
        print('iqr: ', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)  # 하계
        upper_bound = quartile_3 + (iqr * 1.5)  # 상계
        print('lower_bound: ', lower_bound)
        print('upper_bound: ', upper_bound)

        a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound)) 
        allout.append(a)

outliers_loc = outliers(aaa)
print("outlier at :", outliers_loc) 

# box ploting 
# import matplotlib.pyplot as plt

# plt.boxplot(aaa)
# plt.show()