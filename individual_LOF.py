# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.options.display.float_format = '{:.2f}'.format
# %%
ind = pd.read_csv("clean_individuals.csv")

# ---------------------------------------------------------------------------
# CLEANING AND DATA PROCESSING ----------------------------------------------
# ---------------------------------------------------------------------------

# %%
print(ind.dtypes)
print(ind.info)
print(ind.columns)
print(ind.head())
print(ind.dtypes)

# %%
# Changing variable types 
ind["customer_id"] = ind["customer_id"].astype("category")
ind["label"] = ind["label"].astype("category")
ind["country"] = ind["country"].astype("category")
ind["province"] = ind["province"].astype("category")
ind["city"] = ind["city"].astype("category")
ind["gender"] = ind["gender"].astype("category")
ind["marital_status"] = ind["marital_status"].astype("category")
ind["occupation_code"] = ind["occupation_code"].astype("category")

ind["birth_date"] = pd.to_datetime(ind["birth_date"])
ind["onboard_date"] = pd.to_datetime(ind["onboard_date"])

print(ind.dtypes)
# %%
# Inspecting after variable type conversion
ind.describe().style.format("{:.2f}")
ind_summary = ind.describe()

# %%
# Filtering columns with continuous values only
ind_continuous = ind.select_dtypes(include=['int64', 'float64'])

# %%
# Need to drop nulls
ind_continuous = ind_continuous.dropna()

# %%
# Scaling using median and IQR
scaler = RobustScaler()
ind_cont_scaled = scaler.fit_transform(ind_continuous)

# %%
# Converting back into pandas dataframe
ind_cont_scaled = pd.DataFrame(
    ind_cont_scaled,
    columns=ind_continuous.columns,
    index=ind_continuous.index
)

ind_cont_scaled.describe()
# %%
# Selecting columns with low variance
std_after = ind_cont_scaled.std(axis=0)

good_var_cols = ind_cont_scaled.columns[std_after > 0.1]

# %%
# Dropping columns with low variance
ind_LOF_data = ind_cont_scaled[good_var_cols]

'''
# PCA as secondary source for identifying redundant columns
pca = PCA()
ind_pca = pca.fit_transform(ind_cont_scaled)
'''



# %%
from sklearn.neighbors import LocalOutlierFactor

ind_lof = LocalOutlierFactor(
    n_neighbors = 215,
    contamination = "auto" 
)

labels = ind_lof.fit_predict(ind_LOF_data)
scores = pd.Series(
    ind_lof.negative_outlier_factor_,
    index=ind_LOF_data.index
)

# %%
# Score distribution
scores.describe()

# %%
#Top anomalies
anomalies = scores.nsmallest(50)
ind.loc[scores.index, 'lof_score'] = scores

top_anomalies = ind.loc[anomalies.index]
# %%
