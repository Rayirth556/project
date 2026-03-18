import traceback
import sys
import numpy as np
sys.path.insert(0, '.')
from preprocess_real_world_data import *

df = load_home_credit(200)
row = df[df['TARGET']==1].iloc[0]

try:
    v = engineer_features_for_row(row, np.random.default_rng(42))
    print("SUCCESS")
    print(v)
except Exception as e:
    traceback.print_exc()
