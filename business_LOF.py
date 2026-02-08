# %%
from data_preprocessing import cast_categorical
from data_preprocessing import build_debit_credit_features

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
bsn = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/clean_businesses.csv")

# Changing variable types
cast_categorical(bsn)

bsn["established_date"] = pd.to_datetime(bsn["established_date"])
bsn["onboard_date"] = pd.to_datetime(bsn["onboard_date"])
bsn.dtypes

# %%
# Adding new features
channels = ["wire", "eft", "emt", "west", "cheque", "card", "abm"]
bsn_new = bsn.copy()


for ch in channels:
    bsn_new = build_debit_credit_features(bsn_new, ch)

bsn_new = bsn_new.set_index("customer_id")

cols = (
    [c for c in bsn_new.columns if c.endswith("net_transactions")]
    + [c for c in bsn_new.columns if c.endswith("net_amount")]
)

bsn_net = bsn_new[cols].copy()

# %%
# ---------------------------------------------------------------------------
# CLEANING AND DATA PROCESSING ----------------------------------------------
# ---------------------------------------------------------------------------
bsn_net = bsn_net.dropna()

# Scaling using median and IQR
scaler = RobustScaler()
bsn_net_scaled = scaler.fit_transform(bsn_net)

# %%
# Converting back into pandas dataframe
bsn_net_scaled = pd.DataFrame(
    bsn_net_scaled,
    columns=bsn_net.columns,
    index=bsn_net.index
)

bsn_net_scaled.describe()

# %%
# Selecting columns with low variance
std_after = bsn_net_scaled.std(axis=0)

good_var_cols = bsn_net_scaled.columns[std_after < 0.1]
#west_net_transactions
#west_net_amoubt both 
# %%
bsn_net_scaled_subset = bsn_net_scaled.copy()

zero_var_cols = ["west_net_transactions", "west_net_amount"]

bsn_net_scaled_subset = bsn_net_scaled_subset.drop(zero_var_cols, axis=1)



# %%
from sklearn.neighbors import LocalOutlierFactor
# %%
bsn_net_lof = LocalOutlierFactor(
    n_neighbors = 75,
    contamination = "auto" 
)

labels = bsn_net_lof.fit_predict(bsn_net_scaled_subset)
scores = pd.Series(
    bsn_net_lof.negative_outlier_factor_,
    index=bsn_net_scaled_subset.index
)

# %%
# Score distribution
scores.describe()

# %%
#Top anomalies
bsn_new.loc[scores.index, "lof_score"] = scores

anomalies = scores.sort_values(ascending=True)
top_anomalies = bsn_new.loc[anomalies.index]
top_anomalies.to_csv("bsn_net_anomalies.csv")
# %%
# Loggin model info
run_info = {
    "model": type(bsn_net_lof).__name__,
    "n_neighbors": bsn_net_lof.n_neighbors,
    "contamination": bsn_net_lof.contamination,
    "n_samples": bsn_net_scaled_subset.shape[0],
    "n_features": bsn_net_scaled_subset.shape[1],
    "features": list(bsn_net_scaled_subset.columns),
    "score_summary": scores.describe().to_dict()
}

log_run(run_info)
# %%
