import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor

train = pd.read_csv('../data/train_filled.csv')
test = pd.read_csv('../data/test_filled.csv')
model = LGBMRegressor()