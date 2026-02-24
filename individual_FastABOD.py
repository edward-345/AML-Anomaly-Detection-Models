# %%
from pyod.models.abod import ABOD
# --------------------------------------------------------------
# Dependecies
# --------------------------------------------------------------
from pandas.api.types import is_numeric_dtype
from exp_logging import log_run

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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
    "/Users/dangernoodle_/Desktop/DATA/DataTables/master_features_with_clustering.csv")
# %%
# Changing variable types
account, report = cast_binary_and_categorical(account_raw)
print(account.dtypes)

account.dtypes

# Catching stragglers
force_numeric = [
    "months_active",
    "channels_used_count",
    "behavioral_risk_score",
    "txn_score_count_above_threshold"
]

for c in force_numeric:
    account[c] = account[c].astype("float64")


account = account.set_index("customer_id")
acnt_continuous = account.select_dtypes(include=[np.number])

acnt_continuous.shape
acnt_continuous.dtypes


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
    index=acnt_continuous.index
)

acnt_cln_scaled = pd.DataFrame(
    scaler.fit_transform(acnt_continuous_cleaned),
    columns=acnt_continuous_cleaned.columns,
    index=acnt_continuous_cleaned.index
)
# ---------------------------------------------------------------------------
# CHECKING FOR HIGH CORRELATION COLUMNS -------------------------------------
# ---------------------------------------------------------------------------

# %%
corr = acnt_scaled.corr(method="pearson")

# Keep only upper triangle (avoid duplicate pairs)
upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

# Find columns to drop
threshold = 0.7
to_drop = [col for col in upper.columns if (upper[col].abs() >= threshold).any()]


# %%
# Drop them
acnt_reduced = acnt_scaled.drop(columns=to_drop)

corr = acnt_reduced.corr(method="pearson")


# %%
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
plt.figure(figsize=(8, 6))


sns.heatmap(corr, 
            annot=False,
            cmap='coolwarm',
            linewidths=0,
            linecolor='black')

plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------------------------------------
# Rank Tracker ----------------------------------------------------
# ---------------------------------------------------------------------------
# k sweep (FIX: fit on acnt_cln_scl_pca, use cleaned index + filenames)
ks = list(range(20, 100, 10))

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
    "SYNID0200441116", #true outlier from this point below
    "SYNID0103912349"
    #"SYNID0108560369",
   # "SYNID0109015075"
]

rank_by_k_cln = {}

for k in ks:
    abod = ABOD(contamination=0.1,
               method="fast",
               n_neighbors=k)
    _ = abod.fit(acnt_reduced)

    abod_scores = pd.Series(
        abod.decision_scores_,
        index=acnt_reduced.index,
        name="abod_score"
    )

    # ranks for movement plot (1 = most outlier)
    rank_by_k_cln[k] = abod_scores.rank(method="min", ascending=False).astype(int)

    # export pattern
    account_abod = account.copy()
    account_abod.loc[abod_scores.index, "abod_score"] = abod_scores

    top_abod_anomalies = account_abod.nlargest(TOP_N, "abod_score")
    top_abod_anomalies.to_csv(f"accounts_abod_top{TOP_N}_k{k}.csv")


# %%
# ----- movement table + plot
ranks_df_cln = pd.DataFrame(rank_by_k_cln)

tracked_cln = (
    ranks_df_cln.loc[ranks_df_cln.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked_cln["k"] = tracked_cln["k"].astype(int)
tracked_cln = tracked_cln.sort_values(["customer_id", "k"])
tracked_cln.to_csv("rank_ABOD_CLEAN.csv", index=False)

plt.figure()
for cid, g in tracked_cln.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.title("ABOD (CLEAN): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %%
# ---------------------------------------------------------------------------
# ABOD MODEL with k = 60
# ---------------------------------------------------------------------------

abod_model = ABOD(
    contamination=0.1,
    method="fast",
    n_neighbors=60
    )

_ = abod_model.fit(acnt_reduced)
# %%
abod_scores = pd.Series(
    abod_model.decision_scores_,
    index=acnt_reduced.index,
    name="abod_score_raw"
)

# Robust min-max scaling to [0,1]
p1 = abod_scores.quantile(0.01)
p99 = abod_scores.quantile(0.99)
abod_scaled = ((abod_scores - p1) / (p99 - p1)).clip(0, 1)

accounts_ABOD_k60 = pd.DataFrame({
    "customer_id": abod_scores.index,
    "abod_score_01": abod_scaled
}).sort_values("abod_score_01", ascending=False)

accounts_ABOD_k60.to_csv(
    "acc_ABOD_k60_scores_pct.csv",
    index=False,
    float_format="%.6f"
)
# %%
plt.figure()
plt.hist(accounts_ABOD_k60["abod_score_01"], bins=50, edgecolor="black")
plt.xlabel("Outlier Score ABOD")
plt.ylabel("Count")
plt.title("Distribution of ABOD Scores")
plt.grid(True)
plt.show()
# %%
