# %%
# --------------------------------------------------------------
# Dependecies
# --------------------------------------------------------------
from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
import numpy as np

from data_preprocessing import normalize_to_unit_interval

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# %%
# --------------------------------------------------------------
# Pulling Data
# --------------------------------------------------------------
individuals_vA = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/individuals_vA.csv")
ind_VA_FAMD = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/VersionA_Individual_AfterFAMD_V2?.csv")

# 1) Hard checks
if len(individuals_vA) != len(ind_VA_FAMD):
    raise ValueError(f"Row count mismatch: customers={len(individuals_vA)} famd={len(ind_VA_FAMD)}")

if "customer_id" not in individuals_vA.columns:
    raise ValueError("customers CSV must contain customer_id")

# 2) Attach by position safely
indVA_FAMD_output = ind_VA_FAMD.copy()
indVA_FAMD_output.insert(0, "customer_id", individuals_vA["customer_id"].to_numpy())
indVA_FAMD_output.insert(1, "cluster", individuals_vA["cluster"].to_numpy())

# 3) Optional: assert uniqueness
if indVA_FAMD_output["customer_id"].duplicated().any():
    raise ValueError("customer_id is not unique in customers_business_only.csv (unexpected for a key).")

indVA_FAMD_output.to_csv("famd_with_customer_id.csv", index=False)

# %%
# --------------------------------------------------------------
# MAKING SUBSETS BY CLUSTER GROUPING
# --------------------------------------------------------------

# CLUSTER 8 BC-BASED ACTIVE CUSTOMERS
ind8_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_8"]

# Dropping cluster type column
ind8_vAF = ind8_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind8_vAF_copy = ind8_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind8_vAF = ind8_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind8_vAF.reset_index() to put back

# %%
# --------------------------------------------------------------
# LOCAL OUTLIER FACTOR first trial
# --------------------------------------------------------------

ind8_lof_model = LocalOutlierFactor(
    n_neighbors=64,
    contamination="auto",
    n_jobs=-1
    )

ind8_lof_fitted = ind8_lof_model.fit_predict(ind8_vAF)

ind8_lof_scores = pd.Series(
    ind8_lof_model.negative_outlier_factor_,
    index=ind8_vAF.index)

ind8_scored = ind8_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind8_scored.loc[ind8_lof_scores.index, "lof_score_raw"] = ind8_lof_scores

lof_all_norm = normalize_to_unit_interval(ind8_lof_scores)

df_out_ind8 = (
    pd.DataFrame({
        "customer_id": ind8_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind8_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind8 = df_out_ind8[["customer_id", "outlier_score_01"]]

df_out_ind8.to_csv("ind8_LOF_k64_score.csv", index=False,
              float_format="%.10f")

top_ids = ind8_lof_scores.sort_values(ascending=True).index
top_anomalies_ind8 = ind8_scored.loc[top_ids].copy()
top_anomalies_ind8.to_csv("ind8_LOF_k64.csv", index=False)

# %%
# --------------------------------------------------------------
# LOCAL OUTLIER FACTOR LOOP
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "within_cluster_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ks = list(range(60, 70, 1))

TOP_N = 4047

ids_to_track = [
 "SYNID0104333210",
"SYNID0100396047",
"SYNID0104595662",
"SYNID0107588317",
"SYNID0103203861",
"SYNID0100322061",
"SYNID0102198270",
"SYNID0108456366",
"SYNID0106173648",
"SYNID0103486554",
"SYNID0107464935" # LABELED
]

rank_by_k = {}

for k in ks:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind8_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=ind8_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind8 table that has customer_id as a column
    ind8_scored_k = ind8_vAF_copy.copy()
    ind8_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N).index
    top_anomalies_k = ind8_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"ind8_LOF_top{TOP_N}_k{k}.csv"),
        index=False
    )

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind8_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(100, 0) #to change sclae of y-axis
plt.title("LOF (ind8): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# MAKING SUBSETS BY CLUSTER GROUPING
# --------------------------------------------------------------

# CLUSTER 7 --------------------------------------------------------------
ind7_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_7"]

# Dropping cluster type column
ind7_vAF = ind7_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind7_vAF_copy = ind7_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind7_vAF = ind7_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind7_vAF.reset_index() to put back

# CLUSTER 6 --------------------------------------------------------------
ind6_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_6"]

# Dropping cluster type column
ind6_vAF = ind6_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind6_vAF_copy = ind6_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind6_vAF = ind6_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind6_vAF.reset_index() to put back

# CLUSTER 5 --------------------------------------------------------------
ind5_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_5"]

# Dropping cluster type column
ind5_vAF = ind5_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind5_vAF_copy = ind5_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind5_vAF = ind5_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind5_vAF.reset_index() to put back

# CLUSTER 4 --------------------------------------------------------------
ind4_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_4"]

# Dropping cluster type column
ind4_vAF = ind4_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind4_vAF_copy = ind4_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind4_vAF = ind4_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind4_vAF.reset_index() to put back

# CLUSTER 3 -------------------------------------------------------------
ind3_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_3"]

# Dropping cluster type column
ind3_vAF = ind3_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind3_vAF_copy = ind3_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind3_vAF = ind3_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind3_vAF.reset_index() to put back

# CLUSTER 2 -------------------------------------------------------------
ind2_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_2"]

# Dropping cluster type column
ind2_vAF = ind2_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind2_vAF_copy = ind2_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind2_vAF = ind2_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind2_vAF.reset_index() to put back

# CLUSTER 1 -------------------------------------------------------------
ind1_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_1"]

# Dropping cluster type column
ind1_vAF = ind1_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind1_vAF_copy = ind1_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind1_vAF = ind1_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind1_vAF.reset_index() to put back

# CLUSTER 0 -------------------------------------------------------------
ind0_vAF = indVA_FAMD_output[indVA_FAMD_output["cluster"] == "individual_0"]

# Dropping cluster type column
ind0_vAF = ind0_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
ind0_vAF_copy = ind0_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
ind0_vAF = ind0_vAF.set_index("customer_id", drop=True)
#dfind8_vAF = ind0_vAF.reset_index() to put back


# %%
# --------------------------------------------------------------
# CLUSTER 7 FIRST FIT
# --------------------------------------------------------------
# 723 rows
ind7_lof_model = LocalOutlierFactor(
    n_neighbors=27,
    contamination="auto",
    n_jobs=-1
    )

ind7_lof_fitted = ind7_lof_model.fit_predict(ind7_vAF)

ind7_lof_scores = pd.Series(
    ind7_lof_model.negative_outlier_factor_,
    index=ind7_vAF.index)

ind7_scored = ind7_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind7_scored.loc[ind7_lof_scores.index, "lof_score_raw"] = ind7_lof_scores

lof_all_norm = normalize_to_unit_interval(ind7_lof_scores)

df_out_ind7 = (
    pd.DataFrame({
        "customer_id": ind7_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind7_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind7 = df_out_ind7[["customer_id", "outlier_score_01"]]

df_out_ind7.to_csv("ind7_LOF_k27_score.csv", index=False,
              float_format="%.10f")

top_ids = ind7_lof_scores.sort_values(ascending=True).index
top_anomalies_ind7 = ind7_scored.loc[top_ids].copy()
top_anomalies_ind7.to_csv("ind7_LOF_k27.csv", index=False)


pd.read_csv("ind7_LOF_k27.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 7 LOOP
# --------------------------------------------------------------
k7 = list(range(15, 35, 1))

TOP_N_k7 = 723

ids_to_track = [
 "SYNID0100538785",
"SYNID0105390822",
"SYNID0102827801",
"SYNID0109544166",
"SYNID0106761806",
"SYNID0100405376",
"SYNID0100573583",
"SYNID0105286236",
"SYNID0105477924",
"SYNID0108259070"
]

rank_by_k = {}

for k in k7:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind7_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=ind7_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind7 table that has customer_id as a column
    ind7_scored_k = ind7_vAF_copy.copy()
    ind7_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k7).index
    top_anomalies_k = ind7_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"ind7_LOF_top{TOP_N_k7}_k{k}.csv"),
        index=False
    )

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind7_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.ylim(40, 0) #to change sclae of y-axis
plt.title("LOF (ind7): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %%
# --------------------------------------------------------------
# CLUSTER 6 FIRST FIT
# --------------------------------------------------------------
# 367 rows
ind6_lof_model = LocalOutlierFactor(
    n_neighbors=49,
    contamination="auto",
    n_jobs=-1
    )

ind6_lof_fitted = ind6_lof_model.fit_predict(ind6_vAF)

ind6_lof_scores = pd.Series(
    ind6_lof_model.negative_outlier_factor_,
    index=ind6_vAF.index)

ind6_scored = ind6_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind6_scored.loc[ind6_lof_scores.index, "lof_score_raw"] = ind6_lof_scores

lof_all_norm = normalize_to_unit_interval(ind6_lof_scores)

df_out_ind6 = (
    pd.DataFrame({
        "customer_id": ind6_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind6_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind6 = df_out_ind6[["customer_id", "outlier_score_01"]]

df_out_ind6.to_csv("ind6_LOF_k49_score.csv", index=False,
              float_format="%.10f")

top_ids = ind6_lof_scores.sort_values(ascending=True).index
top_anomalies_ind6 = ind6_scored.loc[top_ids].copy()
top_anomalies_ind6.to_csv("ind6_LOF_k49.csv", index=False)


pd.read_csv("ind6_LOF_k49.csv").head()

pd.read_csv("ind6_LOF_k49.csv").tail()
# %%
# --------------------------------------------------------------
# CLUSTER 6 LOOP
# --------------------------------------------------------------
k6 = list(range(20, 70, 1))

TOP_N_k6 = 367

ids_to_track = [
 "SYNID0102233582",
"SYNID0103310035",
"SYNID0102981208",
"SYNID0102385312",
"SYNID0103818734",
"SYNID0107057062",
"SYNID0101926476",
"SYNID0104607741",
"SYNID0104389744",
"SYNID0102494390"
]

rank_by_k = {}

for k in k6:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind6_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=ind6_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind6 table that has customer_id as a column
    ind6_scored_k = ind6_vAF_copy.copy()
    ind6_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k6).index
    top_anomalies_k = ind6_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"ind6_LOF_top{TOP_N_k6}_k{k}.csv"),
        index=False
    )

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind6_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.ylim(30, 0) #to change sclae of y-axis
plt.title("LOF (ind6): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %%
# --------------------------------------------------------------
# CLUSTER 5 FIRST FIT
# --------------------------------------------------------------
# 4392 rows
ind5_lof_model = LocalOutlierFactor(
    n_neighbors=75,
    contamination="auto",
    n_jobs=-1
    )

ind5_lof_fitted = ind5_lof_model.fit_predict(ind5_vAF)

ind5_lof_scores = pd.Series(
    ind5_lof_model.negative_outlier_factor_,
    index=ind5_vAF.index)

ind5_scored = ind5_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind5_scored.loc[ind5_lof_scores.index, "lof_score_raw"] = ind5_lof_scores

lof_all_norm = normalize_to_unit_interval(ind5_lof_scores)

df_out_ind5 = (
    pd.DataFrame({
        "customer_id": ind5_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind5_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind5 = df_out_ind5[["customer_id", "outlier_score_01"]]

df_out_ind5.to_csv("ind5_LOF_k75_score.csv", index=False,
              float_format="%.10f")

top_ids = ind5_lof_scores.sort_values(ascending=True).index
top_anomalies_ind5 = ind5_scored.loc[top_ids].copy()
top_anomalies_ind5.to_csv("ind5_LOF_k75.csv", index=False)


pd.read_csv("ind5_LOF_k75.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 5 LOOP
# --------------------------------------------------------------
k5 = list(range(50, 100, 2))

TOP_N_k5 = 4392 

ids_to_track = [
 "SYNID0107440063",
"SYNID0108981133",
"SYNID0106247241",
"SYNID0106474790",
"SYNID0106989637",
"SYNID0103945410",
"SYNID0103310298",
"SYNID0109441417",
"SYNID0109615266",
"SYNID0103386914"
]

rank_by_k = {}

for k in k5:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind5_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=ind5_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind5 table that has customer_id as a column
    ind5_scored_k = ind5_vAF_copy.copy()
    ind5_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k5).index
    top_anomalies_k = ind5_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"ind5_LOF_top{TOP_N_k5}_k{k}.csv"),
        index=False
    )

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind5_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.ylim(50, 0) #to change sclae of y-axis
plt.title("LOF (ind5): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# CLUSTER 4 FIRST FIT
# --------------------------------------------------------------
# 585 rows
ind4_lof_model = LocalOutlierFactor(
    n_neighbors=22,
    contamination="auto",
    n_jobs=-1
    )

ind4_lof_fitted = ind4_lof_model.fit_predict(ind4_vAF)

ind4_lof_scores = pd.Series(
    ind4_lof_model.negative_outlier_factor_,
    index=ind4_vAF.index)

ind4_scored = ind4_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind4_scored.loc[ind4_lof_scores.index, "lof_score_raw"] = ind4_lof_scores

lof_all_norm = normalize_to_unit_interval(ind4_lof_scores)

df_out_ind4 = (
    pd.DataFrame({
        "customer_id": ind4_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind4_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind4 = df_out_ind4[["customer_id", "outlier_score_01"]]

df_out_ind4.to_csv("ind4_LOF_k22_score.csv", index=False,
              float_format="%.10f")

top_ids = ind4_lof_scores.sort_values(ascending=True).index
top_anomalies_ind4 = ind4_scored.loc[top_ids].copy()
top_anomalies_ind4.to_csv("ind4_LOF_k22.csv", index=False)


pd.read_csv("ind4_LOF_k22.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 4 LOOP
# --------------------------------------------------------------
k4 = list(range(15, 35, 1))

TOP_N_k4 = 585

ids_to_track = [
 "SYNID0105019902",
"SYNID0101391406",
"SYNID0103982184",
"SYNID0102382798",
"SYNID0109600931",
"SYNID0100512696",
"SYNID0100588868",
"SYNID0103739810",
"SYNID0105569155",
"SYNID0103001705"
]

rank_by_k = {}

for k in k4:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind4_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=ind4_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind4 table that has customer_id as a column
    ind4_scored_k = ind4_vAF_copy.copy()
    ind4_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k4).index
    top_anomalies_k = ind4_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"ind4_LOF_top{TOP_N_k4}_k{k}.csv"),
        index=False
    )

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind4_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.ylim(50, 0) #to change sclae of y-axis
plt.title("LOF (ind4): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()



# %%
# --------------------------------------------------------------
# CLUSTER 3 FIRST FIT
# --------------------------------------------------------------
# 15399 rows
ind3_lof_model = LocalOutlierFactor(
    n_neighbors=116,
    contamination="auto",
    n_jobs=-1
    )

ind3_lof_fitted = ind3_lof_model.fit_predict(ind3_vAF)

ind3_lof_scores = pd.Series(
    ind3_lof_model.negative_outlier_factor_,
    index=ind3_vAF.index
)

ind3_scored = ind3_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind3_scored.loc[ind3_lof_scores.index, "lof_score_raw"] = ind3_lof_scores

lof_all_norm = normalize_to_unit_interval(ind3_lof_scores)

df_out_ind3 = (
    pd.DataFrame({
        "customer_id": ind3_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind3_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind3 = df_out_ind3[["customer_id", "outlier_score_01"]]

df_out_ind3.to_csv("ind3_LOF_k116_score.csv", index=False,
              float_format="%.10f")

top_ids = ind3_lof_scores.sort_values(ascending=True).index
top_anomalies_ind3 = ind3_scored.loc[top_ids].copy()
top_anomalies_ind3.to_csv("ind3_LOF_k116.csv", index=False)

pd.read_csv("ind3_LOF_k116.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 3 LOOP
# --------------------------------------------------------------
k3 = list(range(100, 130, 1))

TOP_N_k3 = 15399

ids_to_track = [
 "SYNID0108536321",
"SYNID0101001671",
"SYNID0102242945",
"SYNID0101705003",
"SYNID0109170845",
"SYNID0104358145",
"SYNID0100495947",
"SYNID0107145287",
"SYNID0107235393",
"SYNID0105514169",
"SYNID0100957188" # LABEL
]

rank_by_k = {}

for k in k3:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind3_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=ind3_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind3 table that has customer_id as a column
    ind3_scored_k = ind3_vAF_copy.copy()
    ind3_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k3).index
    top_anomalies_k = ind3_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"ind3_LOF_top{TOP_N_k3}_k{k}.csv"),
    index=False
)

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind3_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(100, 0)
plt.title("LOF (IND3 n = 15399): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
# %%
# --------------------------------------------------------------
# CLUSTER 2 FIRST FIT
# --------------------------------------------------------------
# 1402 rows
ind2_lof_model = LocalOutlierFactor(
    n_neighbors=33,
    contamination="auto",
    n_jobs=-1
    )

ind2_lof_fitted = ind2_lof_model.fit_predict(ind2_vAF)

ind2_lof_scores = pd.Series(
    ind2_lof_model.negative_outlier_factor_,
    index=ind2_vAF.index
)

ind2_scored = ind2_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind2_scored.loc[ind2_lof_scores.index, "lof_score_raw"] = ind2_lof_scores

lof_all_norm = normalize_to_unit_interval(ind2_lof_scores)

df_out_ind2 = (
    pd.DataFrame({
        "customer_id": ind2_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind2_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind2 = df_out_ind2[["customer_id", "outlier_score_01"]]

df_out_ind2.to_csv("ind2_LOF_k33_score.csv", index=False,
              float_format="%.10f")

top_ids = ind2_lof_scores.sort_values(ascending=True).index
top_anomalies_ind2 = ind2_scored.loc[top_ids].copy()
top_anomalies_ind2.to_csv("ind2_LOF_top.csv", index=False)

pd.read_csv("ind2_LOF_top.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 2 LOOP
# --------------------------------------------------------------
k2 = list(range(25, 50, 1))

TOP_N_k2 = 1402

ids_to_track = [
 "SYNID0109335788",
"SYNID0104720263",
"SYNID0105595584",
"SYNID0104884444",
"SYNID0107002354",
"SYNID0103623997",
"SYNID0101296943",
"SYNID0109346581",
"SYNID0107210435",
"SYNID0104781670"
]

rank_by_k = {}

for k in k2:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind2_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=ind2_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind2 table that has customer_id as a column
    ind2_scored_k = ind2_vAF_copy.copy()
    ind2_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k2).index
    top_anomalies_k = ind2_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"ind2_LOF_top{TOP_N_k2}_k{k}.csv"),
    index=False
)

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind2_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.ylim(20, 0)
plt.title("LOF (ind2): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# CLUSTER 1 FIRST FIT
# --------------------------------------------------------------
# 14516 rows
ind1_lof_model = LocalOutlierFactor(
    n_neighbors=120,
    contamination="auto",
    n_jobs=-1
    )

ind1_lof_fitted = ind1_lof_model.fit_predict(ind1_vAF)

ind1_lof_scores = pd.Series(
    ind1_lof_model.negative_outlier_factor_,
    index=ind1_vAF.index
)

ind1_scored = ind1_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind1_scored.loc[ind1_lof_scores.index, "lof_score_raw"] = ind1_lof_scores

lof_all_norm = normalize_to_unit_interval(ind1_lof_scores)

df_out_ind1 = (
    pd.DataFrame({
        "customer_id": ind1_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind1_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind1 = df_out_ind1[["customer_id", "outlier_score_01"]]

df_out_ind1.to_csv("ind1_LOF_k120_scores.csv", index=False,
              float_format="%.10f")

top_ids = ind1_lof_scores.sort_values(ascending=True).index
top_anomalies_ind1 = ind1_scored.loc[top_ids].copy()
top_anomalies_ind1.to_csv("ind1_LOF_top.csv", index=False)

pd.read_csv("ind1_LOF_top.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 1 LOOP
# --------------------------------------------------------------
k1 = list(range(100, 150, 5))

TOP_N_k1 = 14516

ids_to_track = [
  "SYNID0108245672",
"SYNID0106387564",
"SYNID0103464058",
"SYNID0107312487",
"SYNID0105508342",
"SYNID0108264593",
"SYNID0100128464",
"SYNID0109317440",
"SYNID0106113529",
"SYNID0104726828",
"SYNID0101421130", # LABELED
"SYNID0107832828"  # LABELED
]

rank_by_k = {}

for k in k1:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind1_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=ind1_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind1 table that has customer_id as a column
    ind1_scored_k = ind1_vAF_copy.copy()
    ind1_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k1).index
    top_anomalies_k = ind1_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"ind1_LOF_top{TOP_N_k1}_k{k}.csv"),
    index=False
)

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind1_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(30, 0)
plt.title("LOF (IND1 n = 14516): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()



# %%
# --------------------------------------------------------------
# CLUSTER 0 FIRST FIT
# --------------------------------------------------------------
# 11668 rows
ind0_lof_model = LocalOutlierFactor(
    n_neighbors=110,
    contamination="auto",
    n_jobs=-1
    )

ind0_lof_fitted = ind0_lof_model.fit_predict(ind0_vAF)

ind0_lof_scores = pd.Series(
    ind0_lof_model.negative_outlier_factor_,
    index=ind0_vAF.index
)

ind0_scored = ind0_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
ind0_scored.loc[ind0_lof_scores.index, "lof_score_raw"] = ind0_lof_scores

lof_all_norm = normalize_to_unit_interval(ind0_lof_scores)

df_out_ind0 = (
    pd.DataFrame({
        "customer_id": ind0_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": ind0_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_ind0 = df_out_ind0[["customer_id", "outlier_score_01"]]

df_out_ind0.to_csv("ind0_LOF_k110_scores.csv", index=False,
              float_format=
              "%.10f")

top_ids = ind0_lof_scores.sort_values(ascending=True).index
top_anomalies_ind0 = ind0_scored.loc[top_ids].copy()
top_anomalies_ind0.to_csv("ind0_LOF_top.csv", index=False)

pd.read_csv("ind0_LOF_top.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 0 LOOP
# --------------------------------------------------------------
k0 = list(range(80, 160, 5))

TOP_N_k0 = 11668

ids_to_track = [
 "SYNID0109964389",
"SYNID0103497969",
"SYNID0102973153",
"SYNID0102505497",
"SYNID0101629312",
"SYNID0108918519",
"SYNID0103336101",
"SYNID0103645123",
"SYNID0104678842",
"SYNID0103216131",
"SYNID0105593361", # LABELED
"SYNID0107334515" # LABELED
]

rank_by_k = {}

for k in k0:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(ind0_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=ind0_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the ind0 table that has customer_id as a column
    ind0_scored_k = ind0_vAF_copy.copy()
    ind0_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k0).index
    top_anomalies_k = ind0_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"ind0_LOF_top{TOP_N_k0}_k{k}.csv"),
    index=False
)

# ----- movement table + plot
ranks_df = pd.DataFrame(rank_by_k)

tracked = (
    ranks_df.loc[ranks_df.index.intersection(ids_to_track)]
    .reset_index(names="customer_id")
    .melt(id_vars="customer_id", var_name="k", value_name="rank")
)
tracked["k"] = tracked["k"].astype(int)
tracked = tracked.sort_values(["customer_id", "k"])

tracked.to_csv(
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_ind0_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(50, 0)
plt.title("LOF (IND0 n = 11668): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %%
# --------------------------------------------------------------
# SCORES ONLY FOR RULE BASED ALGO
# --------------------------------------------------------------

# Combine all raw LOF scores across clusters into one Series
all_lof_scores = pd.concat([
    ind0_lof_scores,
    ind1_lof_scores,
    ind2_lof_scores,
    ind3_lof_scores,
    ind4_lof_scores,
    ind5_lof_scores,
    ind6_lof_scores,
    ind7_lof_scores,
    ind8_lof_scores,
])

# Negate so higher = more anomalous
global_severity = -all_lof_scores

# Robust scaling using 1st and 99th percentile to avoid extreme outlier compression
p1 = global_severity.quantile(0.01)
p99 = global_severity.quantile(0.99)

def compute_global_extremeness(scored_df, lof_scores):
    df_out = scored_df.copy()
    severity = -lof_scores.loc[df_out.index]
    df_out["within_cluster_extremeness"] = ((severity - p1) / (p99 - p1)).clip(0, 1)
    return df_out

top_anomalies_ind0 = compute_global_extremeness(top_anomalies_ind0, ind0_lof_scores)
top_anomalies_ind1 = compute_global_extremeness(top_anomalies_ind1, ind1_lof_scores)
top_anomalies_ind2 = compute_global_extremeness(top_anomalies_ind2, ind2_lof_scores)
top_anomalies_ind3 = compute_global_extremeness(top_anomalies_ind3, ind3_lof_scores)
top_anomalies_ind4 = compute_global_extremeness(top_anomalies_ind4, ind4_lof_scores)
top_anomalies_ind5 = compute_global_extremeness(top_anomalies_ind5, ind5_lof_scores)
top_anomalies_ind6 = compute_global_extremeness(top_anomalies_ind6, ind6_lof_scores)
top_anomalies_ind7 = compute_global_extremeness(top_anomalies_ind7, ind7_lof_scores)
top_anomalies_ind8 = compute_global_extremeness(top_anomalies_ind8, ind8_lof_scores)

# %%
indVA_SCORES = pd.concat([
    top_anomalies_ind0[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind1[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind2[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind3[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind4[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind5[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind6[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind7[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_ind8[["customer_id", "within_cluster_extremeness"]],
], axis=0, ignore_index=True)

indVA_SCORES.to_csv("indVA_SCORES.csv", index=False)
indVA_SCORES.head()
# %%
indVA_SCORES.to_csv("indVA_SCORES.csv", index=False)

# %%
