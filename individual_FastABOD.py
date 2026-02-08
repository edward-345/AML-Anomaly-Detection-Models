# %%
from pyod.models.abod import ABOD

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

good_var_cols = ind_cont_scaled.columns[std_after > 0.2]

# %%
# Dropping columns with low variance
ind_ABOD_data = ind_cont_scaled[good_var_cols]

# %% Removing credit 
drop = ind_ABOD_data.filter(regex='credit_transactions$').columns

ind_ABOD_subset = ind_ABOD_data.drop(columns=drop)

ind_ABOD_subset = ind_ABOD_subset.drop(columns=['emt_debit_sum',
                                                'emt_credit_sum',
                                                'west_debit_transactions',
                                                'abm_debit_transactions'])
# ---------------------------------------------------------------------------
# CHECKING FOR HIGH CORRELATION COLUMNS -------------------------------------
# ---------------------------------------------------------------------------

# %%
corr = ind_ABOD_subset.corr(method="pearson")

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(corr, aspect="auto")

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

fig.colorbar(cax)
plt.tight_layout()
plt.show()

# %%
high_corr = (
    corr.abs()
    .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)

high_corr.head(20).reset_index(name="corr")
# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.figure(figsize=(8, 6))


sns.heatmap(corr, 
            annot=False,
            cmap='coolwarm',
            linewidths=2,
            linecolor='black')

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# FITTING FastABOD MODEL ----------------------------------------------------
# ---------------------------------------------------------------------------
# %%
ind_ABOD = ABOD(contamination='auto',
                method='fast',
                n_neighbors=30)

ind_ABOD_model = ind_ABOD.fit(ind_ABOD_subset)
# %%
results_df = ind_ABOD_subset.copy()
results_df['abod_score'] = ind_ABOD_model.decision_scores_
results_df['is_outlier'] = ind_ABOD_model.labels_

final_table = ind.merge(
    results_df[['abod_score', 'is_outlier']],
    left_index=True,
    right_index=True,
    how='left'
)

final_table.to_csv("FastABOD_outliers.csv")
# %%
