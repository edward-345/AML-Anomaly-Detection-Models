# %%
# --------------------------------------------------------------
# Dependecies
# --------------------------------------------------------------
from data_preprocessing import cast_categorical
from pandas.api.types import is_numeric_dtype
from exp_logging import log_run

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

def cast_binary_and_categorical(
    df: pd.DataFrame,
    *,
    binary_as_category: bool = True,
    object_as_category: bool = True,
    max_unique_for_int_category: int = 12,
    ignore_cols: tuple[str, ...] = ("customer_id",),
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Cast columns to pandas 'category' when they are:
      (a) binary indicators (0/1, optionally with NaNs), or
      (b) object/string columns (optional), or
      (c) integer-coded categorical columns (optional heuristic: low unique count).

    Leaves continuous numeric columns alone.

    Returns:
      (df_cast, report)
    """
    df = df.copy()
    report = {
        "binary_to_category": [],
        "object_to_category": [],
        "int_low_unique_to_category": [],
        "skipped": [],
    }

    for col in df.columns:
        if col in ignore_cols:
            report["skipped"].append((col, "ignored"))
            continue

        s = df[col]

        # ---- (1) object -> category (nominal text like province/city/industry_code) ----
        if object_as_category and s.dtype == "object":
            df[col] = s.astype("category")
            report["object_to_category"].append(col)
            continue

        # ---- (2) numeric checks ----
        if not is_numeric_dtype(s):
            # already category/datetime/bool/etc.
            continue

        # unique non-missing values
        vals = pd.unique(s.dropna())
        # if nothing but NaNs
        if len(vals) == 0:
            report["skipped"].append((col, "all_missing"))
            continue

        # ---- (2a) binary 0/1 (works even if stored as int64/float64) ----
        if binary_as_category:
            # tolerate {0,1} in any numeric dtype (including floats like 0.0/1.0)
            if set(vals).issubset({0, 1, 0.0, 1.0}):
                # keep as category; optionally make it cleaner by using Int64 first
                # (so NaNs stay NaN and 0/1 remain ints)
                df[col] = s.astype("Int64").astype("category")
                report["binary_to_category"].append(col)
                continue

        # ---- (2b) heuristic: integer-coded categorical (LOW unique count) ----
        # This is optional + conservative: only integers (or integer-like floats) with small cardinality.
        if pd.api.types.is_integer_dtype(s):
            nunique = s.nunique(dropna=True)
            if nunique <= max_unique_for_int_category:
                df[col] = s.astype("category")
                report["int_low_unique_to_category"].append(col)
                continue
        else:
            # float column: check if it's integer-like (e.g., 1.0, 2.0, 3.0) and low-unique
            if pd.api.types.is_float_dtype(s):
                # integer-like if all values are close to whole numbers
                int_like = np.all(np.isclose(vals, np.round(vals)))
                if int_like:
                    nunique = s.nunique(dropna=True)
                    if nunique <= max_unique_for_int_category:
                        df[col] = s.round().astype("Int64").astype("category")
                        report["int_low_unique_to_category"].append(col)
                        continue

        # else: leave continuous numeric columns alone

    if verbose:
        print("Cast report:")
        for k, v in report.items():
            print(f"  {k}: {len(v)}")
        if report["binary_to_category"]:
            print("  binary:", report["binary_to_category"][:15], "..." if len(report["binary_to_category"]) > 15 else "")
        if report["object_to_category"]:
            print("  object:", report["object_to_category"][:15], "..." if len(report["object_to_category"]) > 15 else "")
        if report["int_low_unique_to_category"]:
            print("  int_low_unique:", report["int_low_unique_to_category"][:15], "..." if len(report["int_low_unique_to_category"]) > 15 else "")

    return df, report
# %%
# --------------------------------------------------------------
# Pulling Data
# --------------------------------------------------------------
account_raw = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/features_with_cluster.csv")
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
    "txn_score_count_above_threshold"
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
    index=acnt_continuous.index
)

acnt_cln_scaled = pd.DataFrame(
    scaler.fit_transform(acnt_continuous_cleaned),
    columns=acnt_continuous_cleaned.columns,
    index=acnt_continuous_cleaned.index
)
# %%
# --------------------------------------------------------------
# FITTING LOF
# We will try on both acnt_continuous and acnt_continuous_cleaned
# to see what happens and to compare outputs
#
# We start with K = 200
# --------------------------------------------------------------

from sklearn.neighbors import LocalOutlierFactor
# %%
# ACNT_SCALED
acnt_lof = LocalOutlierFactor(n_neighbors=200,
                              contamination="auto",
                              n_jobs=-1)
acnt_lof_labels = acnt_lof.fit_predict(acnt_scaled)

acnt_lof_scores = pd.Series(acnt_lof.negative_outlier_factor_,
                            index=acnt_scaled.index)

account.loc[acnt_lof_scores.index, "lof_score_all"] = acnt_lof_scores

top_anomalies = account.loc[acnt_lof_scores.nsmallest(1000).index]
top_anomalies.to_csv("accounts_full_LOF.csv")

# %%
# ACNT_CLN_SCALED
acntCLN_lof = LocalOutlierFactor(n_neighbors=200,
                                 contamination="auto",
                                 n_jobs=-1)
acntCLN_lof_labels = acntCLN_lof.fit_predict(acnt_cln_scaled)

acntCLN_lof_scores = pd.Series(acntCLN_lof.negative_outlier_factor_,
                               index=acnt_cln_scaled.index)

account.loc[acntCLN_lof_scores.index, "lof_score_clean"] = acntCLN_lof_scores

top_anomalies = account.loc[acntCLN_lof_scores.nsmallest(1000).index]
top_anomalies.to_csv("accountsCLEAN_full_LOF.csv")

# %%
# --------------------------------------------------------------
# FITTING LOF ON PCA OUTPUT
# We will try on both acnt_continuous and acnt_continuous_cleaned
# --------------------------------------------------------------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=.95,
          random_state = 67)

# %%
#PCA on acnt_scaled
acnt_scl_pca = pca.fit_transform(acnt_scaled)

acnt_scaled.shape[1]
acnt_scl_pca.shape[1]
# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
# %%
