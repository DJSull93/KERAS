import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

aaa = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
                [1,2,25,3,4,6,7,8,900,100,100],
                [-251,2,1,3,4,6,7,78,90,100,10201],
                [1,2,222,3,4,6,45,8,90,100,2020],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])

aaa = aaa.transpose()
print(aaa.shape)


def outlier(data_out):
    lis = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
        print("Q1 : ", quartile_1)
        print("Q2 : ", q2)
        print("Q3 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        print("IQR : ", iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print('lower_bound: ', lower_bound)
        print('upper_bound: ', upper_bound)

        m = np.where((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        n = np.count_nonzero((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        lis.append([i+1,'columns', m, 'outlier_num :', n])

    return np.array(lis)

outliers_loc = outlier(aaa)
print("outlier at :\n", outliers_loc) 

# box ploting 
# import matplotlib.pyplot as plt

# plt.boxplot(aaa)
# plt.show()

'''
Q1 :  3.5
Q2 :  7.0
Q3 :  95.0
IQR :  91.5
lower_bound:  -133.75
upper_bound:  232.25
Q1 :  3.5
Q2 :  7.0
Q3 :  62.5
IQR :  59.0
lower_bound:  -85.0
upper_bound:  151.0
Q1 :  2.5
Q2 :  6.0
Q3 :  84.0
IQR :  81.5
lower_bound:  -119.75
upper_bound:  206.25
Q1 :  3.5
Q2 :  8.0
Q3 :  95.0
IQR :  91.5
lower_bound:  -133.75
upper_bound:  232.25
Q1 :  1000.5
Q2 :  4000.0
Q3 :  6500.0
IQR :  5499.5
lower_bound:  -7248.75
upper_bound:  14749.25
outlier at :
 [[1 'columns' (array([ 2, 10], dtype=int64),) 'outlier_num :' 2]
 [2 'columns' (array([8], dtype=int64),) 'outlier_num :' 1]
 [3 'columns' (array([ 0, 10], dtype=int64),) 'outlier_num :' 2]
 [4 'columns' (array([10], dtype=int64),) 'outlier_num :' 1]
 [5 'columns' (array([], dtype=int64),) 'outlier_num :' 0]]
'''