# %%
import data_preprocessing, inspect

print("Loaded file:", data_preprocessing.__file__)
print("Has function:", hasattr(data_preprocessing, "cast_binary_and_categorical"))

print("\n--- First 40 lines of loaded module ---\n")
print("\n".join(inspect.getsource(data_preprocessing).splitlines()[:40]))
print("\n--------------------------------------\n")

# %%
from data_preprocessing import cast_categorical
from data_preprocessing import cast_binary_and_categorical

from exp_logging import log_run

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# %%
account = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/features_with_cluster.csv")
# %%
# Changing variable types
cast_categorical(account)

account.dtypes
# %%
