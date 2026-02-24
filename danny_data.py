# %%
# --------------------------------------------------------------
# Dependecies
# --------------------------------------------------------------
from pandas.api.types import is_numeric_dtype
from exp_logging import log_run

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

from data_preprocessing import cast_binary_and_categorical
from data_preprocessing import normalize_to_unit_interval

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# %%
# --------------------------------------------------------------
# Pulling Data
# --------------------------------------------------------------
account_raw = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/master_features_with_clustering.csv"
)

# %%
# Changing variable types
account, report = cast_binary_and_categorical(account_raw)
print(account.dtypes)
account.dtypes

# %%
# Catching stragglers
force_numeric = [
    "months_active",
    "channels_used_count",
    "behavioral_risk_score",
    "txn_score_count_above_threshold",
]

for c in force_numeric:
    account[c] = account[c].astype("float64")

# %%
account = account.set_index("customer_id")
acnt_continuous = account.select_dtypes(include=[np.number])

acnt_continuous.shape
acnt_continuous.dtypes

# %%
acnt_continuous_cleaned = acnt_continuous.loc[:, acnt_continuous.var() > 0.1]
acnt_continuous_cleaned.shape

# %%
# --------------------------------------------------------------
# SCALING
# We will try on both acnt_continuous and acnt_continuous_cleaned
# to see what happens and to compare outputs
# --------------------------------------------------------------
scaler = RobustScaler()

acnt_scaled = pd.DataFrame(
    scaler.fit_transform(acnt_continuous),
    columns=acnt_continuous.columns,
    index=acnt_continuous.index,
)

acnt_cln_scaled = pd.DataFrame(
    scaler.fit_transform(acnt_continuous_cleaned),
    columns=acnt_continuous_cleaned.columns,
    index=acnt_continuous_cleaned.index,
)

# %%
# --------------------------------------------------------------
# FITTING LOF
# We will try on both acnt_continuous and acnt_continuous_cleaned
# to see what happens and to compare outputs
#
# We start with K = 200
# --------------------------------------------------------------

# %%
# ACNT_SCALED
acnt_lof = LocalOutlierFactor(
    n_neighbors=140,
    contamination="auto",
    n_jobs=-1,
)
acnt_lof_labels = acnt_lof.fit_predict(acnt_scaled)

acnt_lof_scores = pd.Series(
    acnt_lof.negative_outlier_factor_,
    index=acnt_scaled.index,
)

account.loc[acnt_lof_scores.index, "lof_score_all"] = acnt_lof_scores

lof_all_norm = normalize_to_unit_interval(acnt_lof_scores)

df_out = pd.DataFrame(
    {
        "customer_id": lof_all_norm.index,
        "outlier_score_01": lof_all_norm.values,
    }
)

df_out.to_csv(
    "accounts_full_LOF_score_only.csv",
    index=False,
    float_format="%.3f",
)

top_anomalies = account.loc[acnt_lof_scores.nsmallest(1000).index]
top_anomalies.to_csv("accounts_full_LOF.csv")

# %%
# ACNT_CLN_SCALED
acntCLN_lof = LocalOutlierFactor(
    n_neighbors=140,
    contamination="auto",
    n_jobs=-1,
)
acntCLN_lof_labels = acntCLN_lof.fit_predict(acnt_cln_scaled)

acntCLN_lof_scores = pd.Series(
    acntCLN_lof.negative_outlier_factor_,
    index=acnt_cln_scaled.index,
)

account.loc[acntCLN_lof_scores.index, "lof_score_clean"] = acntCLN_lof_scores

lof_clean_norm = normalize_to_unit_interval(acntCLN_lof_scores)

df_out = pd.DataFrame(
    {
        "customer_id": lof_clean_norm.index,
        "outlier_score_01": lof_clean_norm.values,
    }
)

df_out.to_csv(
    "accounts_CLEAN_LOF_score_only.csv",
    index=False,
    float_format="%.3f",
)

top_anomalies = account.loc[acntCLN_lof_scores.nsmallest(1000).index]
top_anomalies.to_csv("accountsCLEAN_full_LOF.csv")

# %%
# --------------------------------------------------------------
# FITTING LOF ON PCA OUTPUT for ACNT full columns
# We will try on both acnt_continuous and acnt_continuous_cleaned
# --------------------------------------------------------------
pca_full = PCA(
    n_components=0.95,
    random_state=67,
)

pd.options.display.float_format = "{:.2f}".format

# %%
# PCA on acnt_scaled
acnt_scl_pca = pca_full.fit_transform(acnt_scaled)

# Number of cols it reduces to
acnt_scl_pca.shape[1]

# %%
# Cumulative variance plot
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure()
plt.plot(cumvar)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

# %%
# scree plot
plt.figure()
plt.plot(pca_full.explained_variance_)
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()

# %%
# looking at components
loadings = pd.DataFrame(
    pca_full.components_,
    columns=acnt_scaled.columns,
)
loadings.iloc[0]

# %%
params = {"n_neighbors": 140, "contamination": "auto"}

acnt_lof_pca = LocalOutlierFactor(**params)
acnt_lof_pca_labels = acnt_lof_pca.fit_predict(acnt_scl_pca)

acnt_lof_pca_scores = pd.Series(
    acnt_lof_pca.negative_outlier_factor_,
    index=acnt_scaled.index,
)

lof_pca_norm = normalize_to_unit_interval(acnt_lof_pca_scores)

df_out = pd.DataFrame(
    {
        "customer_id": lof_pca_norm.index,
        "outlier_score_01": lof_pca_norm.values,
    }
)

df_out.to_csv(
    "accounts_full_PCA_LOF_score_only.csv",
    index=False,
    float_format="%.3f",
)

# %%
account_pca = account.copy()
account_pca.loc[acnt_lof_pca_scores.index, "lof_score_pca"] = acnt_lof_pca_scores

top_pca_anomalies = account_pca.nsmallest(6000, "lof_score_pca")
top_pca_anomalies.to_csv("accounts_full_pca_LOF.csv")

# %%
ks = list(range(150, 300, 5))

TOP_N = 6000
ids_to_track = [
    "SYNID0100957188",
    "SYNID0101421130",
    "SYNID0105593361",
    "SYNID0107334515",
    "SYNID0107464935",
    "SYNID0107832828",
    "SYNID0200187014",
    "SYNID0200496670",
    "SYNID0200755574",
    "SYNID0200755995",
    "SYNID0200441116",  # true outlier from this point below
    "SYNID0103912349",
    # "SYNID0108560369",
    # "SYNID0109015075",
]

rank_by_k = {}

for k in ks:
    lof = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1,
    )
    _ = lof.fit_predict(acnt_scl_pca)

    lof_scores = pd.Series(
        lof.negative_outlier_factor_,
        index=acnt_scaled.index,
        name="lof_score_pca",
    )

    # ranks for movement plot (1 = most outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # your exact export pattern
    account_pca = account.copy()
    account_pca.loc[lof_scores.index, "lof_score_pca"] = lof_scores

    top_pca_anomalies = account_pca.nsmallest(TOP_N, "lof_score_pca")
    top_pca_anomalies.to_csv(f"accounts_full_pca_LOF_top{TOP_N}_k{k}.csv")

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])
tracked.to_csv("tracked_rank_movement_pca.csv", index=False)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.title("PCA-LOF: rank movement across k")
plt.grid(True)
plt.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.show()

# %%
# --------------------------------------------------------------
# FITTING LOF ON PCA OUTPUT for ACNT cleaned
# We will try on both acnt_continuous and acnt_continuous_cleaned
# --------------------------------------------------------------
pd.options.display.float_format = "{:.2f}".format

pca_cln = PCA(
    n_components=0.95,
    random_state=67,
)

# %%
# PCA on acnt_cln_scaled
acnt_cln_scl_pca = pca_cln.fit_transform(acnt_cln_scaled)

# Number of cols it reduces to
acnt_cln_scl_pca.shape[1]

# %%
# Cumulative variance plot
cumvar_cln = np.cumsum(pca_cln.explained_variance_ratio_)

plt.figure()
plt.plot(cumvar_cln)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

# %%
# scree plot
plt.figure()
plt.plot(pca_cln.explained_variance_)
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()

# %%
# looking at components (FIX: use cleaned columns)
loadings_cln = pd.DataFrame(
    pca_cln.components_,
    columns=acnt_continuous_cleaned.columns,
)
loadings_cln.iloc[0]

# %%
params_cln = {"n_neighbors": 140, "contamination": "auto"}

acntCLN_lof_pca = LocalOutlierFactor(**params_cln, n_jobs=-1)
acntCLN_lof_pca_labels = acntCLN_lof_pca.fit_predict(acnt_cln_scl_pca)

# IMPORTANT: index should match accounts (customer_id). Use acnt_cln_scaled.index
acntCLN_lof_pca_scores = pd.Series(
    acntCLN_lof_pca.negative_outlier_factor_,
    index=acnt_cln_scaled.index,
)

lof_pca_clean_norm = normalize_to_unit_interval(acntCLN_lof_pca_scores)

df_out = pd.DataFrame(
    {
        "customer_id": lof_pca_clean_norm.index,
        "outlier_score_01": lof_pca_clean_norm.values,
    }
)

df_out.to_csv(
    "accounts_CLEAN_PCA_LOF_score_only.csv",
    index=False,
    float_format="%.3f",
)

# %%
accountCLN_pca = account.copy()
accountCLN_pca.loc[acntCLN_lof_pca_scores.index, "lof_score_pca"] = acntCLN_lof_pca_scores

top_pca_anomalies_cln = accountCLN_pca.nsmallest(6000, "lof_score_pca")
top_pca_anomalies_cln.to_csv("accounts_CLEAN_pca_LOF.csv")

# %%
# k sweep (FIX: fit on acnt_cln_scl_pca, use cleaned index + filenames)
# ks = list(range(100, 201, 20))

TOP_N = 6000
ids_to_track = [
    "SYNID0100957188",
    "SYNID0101421130",
    "SYNID0105593361",
    "SYNID0107334515",
    "SYNID0107464935",
    "SYNID0107832828",
    "SYNID0200187014",
    "SYNID0200496670",
    "SYNID0200755574",
    "SYNID0200755995",
    "SYNID0200441116",  # true outlier from this point below
    "SYNID0103912349",
    "SYNID0108560369",
    "SYNID0109015075",
]

rank_by_k_cln = {}

for k in ks:
    lof = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1,
    )
    _ = lof.fit_predict(acnt_cln_scl_pca)

    lof_scores = pd.Series(
        lof.negative_outlier_factor_,
        index=acnt_cln_scaled.index,
        name="lof_score_pca",
    )

    # ranks for movement plot (1 = most outlier)
    rank_by_k_cln[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # export pattern
    account_pca = account.copy()
    account_pca.loc[lof_scores.index, "lof_score_pca"] = lof_scores

    top_pca_anomalies = account_pca.nsmallest(TOP_N, "lof_score_pca")
    top_pca_anomalies.to_csv(f"accounts_CLEAN_pca_LOF_top{TOP_N}_k{k}.csv")


ranks_df_cln = pd.DataFrame(rank_by_k_cln)

tracked_cln = (
    ranks_df_cln.loc[ranks_df_cln.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked_cln["k"] = tracked_cln["k"].astype(int)
tracked_cln = tracked_cln.sort_values(["customer_id", "k"])
tracked_cln.to_csv("tracked_rank_movement_pca_CLEAN.csv", index=False)

plt.figure()
for cid, g in tracked_cln.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.title("PCA-LOF (CLEAN): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# FINAL MODEL (PCA-LOF on FULL scaled features)
# --------------------------------------------------------------

FINAL_K = 240

final_lof_model = LocalOutlierFactor(
    n_neighbors=FINAL_K,
    contamination="auto",
    n_jobs=-1,
)

_ = final_lof_model.fit_predict(acnt_scl_pca)

# raw LOF scores (more negative = more outlier)
final_lof_scores = pd.Series(
    final_lof_model.negative_outlier_factor_,
    index=acnt_scaled.index,          # customer_id index
    name="lof_score_pca_raw",
)

# normalize to 0-1 (higher should mean more outlier in your pipeline)
final_lof_norm = normalize_to_unit_interval(final_lof_scores)

# score-only export (2 cols), with tie-resolved ordering
df_out_final = (
    pd.DataFrame({
        "customer_id": final_lof_scores.index,
        "outlier_score_01": final_lof_norm.values,
        "lof_score_raw": final_lof_scores.values,
    })
    .sort_values(
        ["outlier_score_01", "lof_score_raw", "customer_id"],
        ascending=[False, True, True],
        kind="mergesort",  # stable sort
    )
)

df_out_final[["customer_id", "outlier_score_01"]].to_csv(
    f"accounts_full_PCA_LOF_k{FINAL_K}_score_only.csv",
    index=False,
    float_format="%.10f",
)

# full table export: ALL rows, ordered most outlier → least, matching df_out_final order
account_final = account.copy()
account_final.loc[final_lof_scores.index, "lof_score_pca_raw"] = final_lof_scores

final_ranked_ids = df_out_final["customer_id"].tolist()
final_ranked = account_final.loc[final_ranked_ids].copy()

final_ranked.to_csv(
    f"accounts_full_PCA_LOF_k{FINAL_K}_ALL_ranked.csv",
    index=True,  # keeps customer_id index
)
# %%
