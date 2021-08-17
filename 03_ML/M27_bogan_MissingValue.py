# [1, np.nan, np.nan, 8, 10]

# 결측치 처리
# 1. 행 삭제 !!!
# 2. 0 or 특정값 대입 -> [1, 0, 0, 8, 10]
# 3. 전데이터            [1, 1, 1, 8, 10]
# 4. 후데이터            [1, 8, 8, 8, 10]
# 5. 중위값              [1, 4.5, 4.5, 8, 10]
# 6. 보간 (linear) !!!
# 7. 모델링 - predict 
# //: 결측치 제외 나머지 데이터로 x 구성
# //: predict로 결측치 채워줌 
# 8. boost type : 결측치에 대해 자유롭다 (안해도됨)

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ["08/13/2021", "08/14/2021", "08/15/2021", "08/16/2021", "08/17/2021"]
dates = pd.to_datetime(datastrs)

# print(dates) # dtype='datetime64[ns]', freq=None
print("=================================================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
# print(ts)

to_intp_linear = ts.interpolate()
print(to_intp_linear)

'''
2021-08-13     1.000000
2021-08-14     3.333333
2021-08-15     5.666667
2021-08-16     8.000000
2021-08-17    10.000000
'''