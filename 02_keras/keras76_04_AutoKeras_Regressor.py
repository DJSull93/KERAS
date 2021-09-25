from sklearn.datasets import load_boston
import autokeras as ak
import pandas as pd

# 1. data
datasets = load_boston()

x = datasets.data
y = datasets.target


# 2. model
model = ak.StructuredDataRegressor()

# 3. train
model.fit(x, y, epochs=2, validation_split=0.2)

# 4. eval pred
res = model.evaluate(x, y)
print(res)

#############