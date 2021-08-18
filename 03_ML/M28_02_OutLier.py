import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

aaa = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
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
print("outlier at :", outliers_loc) 

# box ploting 
# import matplotlib.pyplot as plt

# plt.boxplot(aaa)
# plt.show()