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

# ---------------------------------------------------------------------------
# LOF MODEL FITTING  --------------------------------------------------------
# ---------------------------------------------------------------------------


# %%
from sklearn.neighbors import LocalOutlierFactor
# %%
ind_lof = LocalOutlierFactor(
    n_neighbors = 350,
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
anomalies = scores.nsmallest(1000)
ind.loc[scores.index, 'lof_score'] = scores

top_anomalies = ind.loc[anomalies.index]
# %%
top_anomalies.to_csv("check_anomalies.csv")



# ---------------------------------------------------------------------------
# LOF FITTING ON PCA OUTPUT -------------------------------------------------
# doesnt rly work yet...
# ---------------------------------------------------------------------------

# %%
# choosing number of components
ind_pca_raw = PCA(n_components=2) 
ind_pca = ind_pca_raw.fit_transform(ind_LOF_data)

 # %%
# 2. Convert back to DataFrame to keep your Index safe
pc_cols = [f'PC{i+1}' for i in range(ind_pca.shape[1])]
ind_pca_df = pd.DataFrame(ind_pca, columns=pc_cols, index=ind_LOF_data.index)

# %%
# 3. Fit LOF on the PCA components
# We use the PCA coordinates here
pca_labels = ind_lof.fit_predict(ind_pca_df)
pca_scores = ind_lof.negative_outlier_factor_

# %%
# 4. Join back to your original 'ind' dataframe
ind_pca_results = ind.copy()

# Map the scores back using the index from the PCA dataframe
ind_pca_results['lof_score_pca'] = pd.Series(pca_scores, index=ind_pca_df.index)

# 5. Filter for the top 1000 anomalies specifically from this PCA run
# Since ind_pca_results contains all rows (including those skipped), 
# we sort by the new score column.
top_pca_anomalies = ind_pca_results.nsmallest(1000, 'lof_score_pca')

top_pca_anomalies.to_csv("anomaly_report_pca.csv", index=True)
# %%
