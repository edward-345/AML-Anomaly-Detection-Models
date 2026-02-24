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
businesses_vA = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/businesses_vA.csv")
bsn_VA_FAMD = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/VersionA_Business_AfterFAMD.csv")

# %%
# 1) Hard checks
if len(businesses_vA) != len(bsn_VA_FAMD):
    raise ValueError(f"Row count mismatch: customers={len(businesses_vA)} famd={len(bsn_VA_FAMD)}")

if "customer_id" not in businesses_vA.columns:
    raise ValueError("customers CSV must contain customer_id")

# 2) Attach by position safely
bsnVA_FAMD_output = bsn_VA_FAMD.copy()
bsnVA_FAMD_output.insert(0, "customer_id", businesses_vA["customer_id"].to_numpy())
bsnVA_FAMD_output.insert(1, "cluster", businesses_vA["cluster"].to_numpy())

# 3) Optional: assert uniqueness
if bsnVA_FAMD_output["customer_id"].duplicated().any():
    raise ValueError("customer_id is not unique in customers_business_only.csv (unexpected for a key).")

bsnVA_FAMD_output.to_csv("famd_with_customer_id.csv", index=False)

# %%
# --------------------------------------------------------------
# MAKING SUBSETS BY CLUSTER GROUPING
# --------------------------------------------------------------

# CLUSTER 6
bsn6_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_6"]

# Dropping cluster type column
bsn6_vAF = bsn6_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn6_vAF_copy = bsn6_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn6_vAF = bsn6_vAF.set_index("customer_id", drop=True)
#dfbsn6_vAF = bsn6_vAF.reset_index() to put back

# %%
# --------------------------------------------------------------
# LOCAL OUTLIER FACTOR first trial
# --------------------------------------------------------------

bsn6_lof_model = LocalOutlierFactor(
    n_neighbors=13,
    contamination="auto",
    n_jobs=-1
    )

bsn6_lof_fitted = bsn6_lof_model.fit_predict(bsn6_vAF)

bsn6_lof_scores = pd.Series(
    bsn6_lof_model.negative_outlier_factor_,
    index=bsn6_vAF.index)

bsn6_scored = bsn6_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn6_scored.loc[bsn6_lof_scores.index, "lof_score_raw"] = bsn6_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn6_lof_scores)

df_out_bsn6 = (
    pd.DataFrame({
        "customer_id": bsn6_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn6_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn6 = df_out_bsn6[["customer_id", "outlier_score_01"]]

df_out_bsn6.to_csv("bsn6_LOF_k13_score.csv", index=False,
              float_format="%.10f")

top_ids = bsn6_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn6 = bsn6_scored.loc[top_ids].copy()
top_anomalies_bsn6.to_csv("bsn6_LOF_k13.csv", index=False)

# %%
# --------------------------------------------------------------
# LOCAL OUTLIER FACTOR LOOP
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "within_cluster_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ks = list(range(5, 30, 1))

TOP_N = 178

ids_to_track = [
    "SYNID0200931678",
"SYNID0200999280",
"SYNID0200159270",
"SYNID0200161910",
"SYNID0200100125",
"SYNID0200318536",
"SYNID0200668364",
"SYNID0200349306",
"SYNID0200791566",
"SYNID0200328575"
]

rank_by_k = {}

for k in ks:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn6_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=bsn6_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn6 table that has customer_id as a column
    bsn6_scored_k = bsn6_vAF_copy.copy()
    bsn6_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N).index
    top_anomalies_k = bsn6_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"bsn6_LOF_top{TOP_N}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn6_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(20, 0) #to change sclae of y-axis
plt.title("LOF (bsn6): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()






# %%
# --------------------------------------------------------------
# MAKING SUBSETS BY CLUSTER GROUPING
# --------------------------------------------------------------

# CLUSTER 5 --------------------------------------------------------------
bsn5_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_5"]

# Dropping cluster type column
bsn5_vAF = bsn5_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn5_vAF_copy = bsn5_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn5_vAF = bsn5_vAF.set_index("customer_id", drop=True)
#dfbsn7_vAF = bsn5_vAF.reset_index() to put back

# CLUSTER 4 --------------------------------------------------------------
bsn4_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_4"]

# Dropping cluster type column
bsn4_vAF = bsn4_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn4_vAF_copy = bsn4_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn4_vAF = bsn4_vAF.set_index("customer_id", drop=True)
#dfbsn7_vAF = bsn4_vAF.reset_index() to put back

# CLUSTER 3 -------------------------------------------------------------
bsn3_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_3"]

# Dropping cluster type column
bsn3_vAF = bsn3_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn3_vAF_copy = bsn3_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn3_vAF = bsn3_vAF.set_index("customer_id", drop=True)
#dfbsn7_vAF = bsn3_vAF.reset_index() to put back

# CLUSTER 2 -------------------------------------------------------------
bsn2_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_2"]

# Dropping cluster type column
bsn2_vAF = bsn2_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn2_vAF_copy = bsn2_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn2_vAF = bsn2_vAF.set_index("customer_id", drop=True)
#dfbsn7_vAF = bsn2_vAF.reset_index() to put back

# CLUSTER 1 -------------------------------------------------------------
bsn1_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_1"]

# Dropping cluster type column
bsn1_vAF = bsn1_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn1_vAF_copy = bsn1_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn1_vAF = bsn1_vAF.set_index("customer_id", drop=True)
#dfbsn7_vAF = bsn1_vAF.reset_index() to put back

# CLUSTER 0 -------------------------------------------------------------
bsn0_vAF = bsnVA_FAMD_output[bsnVA_FAMD_output["cluster"] == "business_0"]

# Dropping cluster type column
bsn0_vAF = bsn0_vAF.drop(columns=["cluster"])

# Keeping customer id as index AND column for reattaching
bsn0_vAF_copy = bsn0_vAF.set_index("customer_id", drop=False)

# Setting customer id as index
bsn0_vAF = bsn0_vAF.set_index("customer_id", drop=True)
#dfbsn7_vAF = bsn0_vAF.reset_index() to put back


# %%
# --------------------------------------------------------------
# CLUSTER 5 FIRST FIT
# --------------------------------------------------------------
# 246 rows
bsn5_lof_model = LocalOutlierFactor(
    n_neighbors=13,
    contamination="auto",
    n_jobs=-1
    )

bsn5_lof_fitted = bsn5_lof_model.fit_predict(bsn5_vAF)

bsn5_lof_scores = pd.Series(
    bsn5_lof_model.negative_outlier_factor_,
    index=bsn5_vAF.index)

bsn5_scored = bsn5_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn5_scored.loc[bsn5_lof_scores.index, "lof_score_raw"] = bsn5_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn5_lof_scores)

df_out_bsn5 = (
    pd.DataFrame({
        "customer_id": bsn5_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn5_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn5 = df_out_bsn5[["customer_id", "outlier_score_01"]]

df_out_bsn5.to_csv("bsn5_LOF_k13_score.csv", index=False,
              float_format="%.10f")

top_ids = bsn5_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn5 = bsn5_scored.loc[top_ids].copy()
top_anomalies_bsn5.to_csv("bsn5_LOF_k13.csv", index=False)


pd.read_csv("bsn5_LOF_k13.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 5 LOOP
# --------------------------------------------------------------
k5 = list(range(10, 16, 1))

TOP_N_k5 = 246

ids_to_track = [
    "SYNID0200451590",
"SYNID0200210370",
"SYNID0200992582",
"SYNID0200961090",
"SYNID0200701399",
"SYNID0200386004",
"SYNID0200594442",
"SYNID0200487583",
"SYNID0200621069",
"SYNID0200700793"
]

rank_by_k = {}

for k in k5:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn5_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=bsn5_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn5 table that has customer_id as a column
    bsn5_scored_k = bsn5_vAF_copy.copy()
    bsn5_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k5).index
    top_anomalies_k = bsn5_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"bsn5_LOF_top{TOP_N_k5}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn5_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
plt.ylim(25, 0) #to change sclae of y-axis
plt.title("LOF (bsn5): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# CLUSTER 4 FIRST FIT
# --------------------------------------------------------------
# 652 rows
bsn4_lof_model = LocalOutlierFactor(
    n_neighbors=25,
    contamination="auto",
    n_jobs=-1
    )

bsn4_lof_fitted = bsn4_lof_model.fit_predict(bsn4_vAF)

bsn4_lof_scores = pd.Series(
    bsn4_lof_model.negative_outlier_factor_,
    index=bsn4_vAF.index)

bsn4_scored = bsn4_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn4_scored.loc[bsn4_lof_scores.index, "lof_score_raw"] = bsn4_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn4_lof_scores)

df_out_bsn4 = (
    pd.DataFrame({
        "customer_id": bsn4_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn4_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn4 = df_out_bsn4[["customer_id", "outlier_score_01"]]

df_out_bsn4.to_csv("bsn4_LOF_k25_score.csv", index=False,
              float_format="%.10f")

top_ids = bsn4_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn4 = bsn4_scored.loc[top_ids].copy()
top_anomalies_bsn4.to_csv("bsn4_LOF_k25.csv", index=False)


pd.read_csv("bsn4_LOF_k25.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 4 LOOP
# --------------------------------------------------------------
k4 = list(range(10, 40, 2))

TOP_N_k4 = 652

ids_to_track = [
"SYNID0200820259",
"SYNID0200803216",
"SYNID0200057370",
"SYNID0200517734",
"SYNID0200502175",
"SYNID0200214714",
"SYNID0200954599",
"SYNID0200628433",
"SYNID0200455319",
"SYNID0200718828",
"SYNID0200187014" # LABELED
]

rank_by_k = {}

for k in k4:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn4_vAF)

    lof_scores = pd.Series(
        lof_loop.negative_outlier_factor_,
        index=bsn4_vAF.index,
        name="lof_score_raw"
    )

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn4 table that has customer_id as a column
    bsn4_scored_k = bsn4_vAF_copy.copy()
    bsn4_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k4).index
    top_anomalies_k = bsn4_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
        os.path.join(OUTPUT_DIR, f"bsn4_LOF_top{TOP_N_k4}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn4_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(50, 0) #to change sclae of y-axis
plt.title("LOF (BSN4 n = 652): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()



# %%
# --------------------------------------------------------------
# CLUSTER 3 FIRST FIT
# --------------------------------------------------------------
# 1616 rows
bsn3_lof_model = LocalOutlierFactor(
    n_neighbors=30,
    contamination="auto",
    n_jobs=-1
    )

bsn3_lof_fitted = bsn3_lof_model.fit_predict(bsn3_vAF)

bsn3_lof_scores = pd.Series(
    bsn3_lof_model.negative_outlier_factor_,
    index=bsn3_vAF.index
)

bsn3_scored = bsn3_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn3_scored.loc[bsn3_lof_scores.index, "lof_score_raw"] = bsn3_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn3_lof_scores)

df_out_bsn3 = (
    pd.DataFrame({
        "customer_id": bsn3_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn3_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn3 = df_out_bsn3[["customer_id", "outlier_score_01"]]

df_out_bsn3.to_csv("bsn3_LOF_k30_score.csv", index=False,
              float_format="%.10f")

top_ids = bsn3_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn3 = bsn3_scored.loc[top_ids].copy()
top_anomalies_bsn3.to_csv("bsn3_LOF_k30.csv", index=False)

pd.read_csv("bsn3_LOF_k30.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 3 LOOP
# --------------------------------------------------------------
k3 = list(range(10, 60, 2))

TOP_N_k3 = 1616

ids_to_track = [
"SYNID0200238787",
"SYNID0200904892",
"SYNID0200510094",
"SYNID0200991381",
"SYNID0200639238",
"SYNID0200470265",
"SYNID0200744171",
"SYNID0200861567",
"SYNID0200689325",
"SYNID0200353288",
"SYNID0200496670", # LABELED
"SYNID0200755995" # LABELED
]

rank_by_k = {}

for k in k3:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn3_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=bsn3_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn3 table that has customer_id as a column
    bsn3_scored_k = bsn3_vAF_copy.copy()
    bsn3_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k3).index
    top_anomalies_k = bsn3_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"bsn3_LOF_top{TOP_N_k3}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn3_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(100, 0)
plt.title("LOF (BSN3 n=1616): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
# %%
# --------------------------------------------------------------
# CLUSTER 2 FIRST FIT
# --------------------------------------------------------------
# 108 rows
bsn2_lof_model = LocalOutlierFactor(
    n_neighbors=10,
    contamination="auto",
    n_jobs=-1
    )

bsn2_lof_fitted = bsn2_lof_model.fit_predict(bsn2_vAF)

bsn2_lof_scores = pd.Series(
    bsn2_lof_model.negative_outlier_factor_,
    index=bsn2_vAF.index
)

bsn2_scored = bsn2_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn2_scored.loc[bsn2_lof_scores.index, "lof_score_raw"] = bsn2_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn2_lof_scores)

df_out_bsn2 = (
    pd.DataFrame({
        "customer_id": bsn2_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn2_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn2 = df_out_bsn2[["customer_id", "outlier_score_01"]]

df_out_bsn2.to_csv("bsn2_LOF_k10_score.csv", index=False,
              float_format="%.10f")

top_ids = bsn2_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn2 = bsn2_scored.loc[top_ids].copy()
top_anomalies_bsn2.to_csv("bsn2_LOF_top.csv", index=False)

pd.read_csv("bsn2_LOF_top.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 2 LOOP
# --------------------------------------------------------------
k2 = list(range(5, 20, 1))

TOP_N_k2 = 108

ids_to_track = [
 "SYNID0200422113",
"SYNID0200832632",
"SYNID0200058125",
"SYNID0200294013",
"SYNID0200342835",
"SYNID0200971324",
"SYNID0200073454",
"SYNID0200381719",
"SYNID0200623060",
"SYNID0200704308"
]

rank_by_k = {}

for k in k2:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn2_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=bsn2_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn2 table that has customer_id as a column
    bsn2_scored_k = bsn2_vAF_copy.copy()
    bsn2_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k2).index
    top_anomalies_k = bsn2_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"bsn2_LOF_top{TOP_N_k2}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn2_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(100, 0)
plt.title("LOF (bsn2): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# CLUSTER 1 FIRST FIT
# --------------------------------------------------------------
# 1194 rows
bsn1_lof_model = LocalOutlierFactor(
    n_neighbors=35,
    contamination="auto",
    n_jobs=-1
    )

bsn1_lof_fitted = bsn1_lof_model.fit_predict(bsn1_vAF)

bsn1_lof_scores = pd.Series(
    bsn1_lof_model.negative_outlier_factor_,
    index=bsn1_vAF.index
)

bsn1_scored = bsn1_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn1_scored.loc[bsn1_lof_scores.index, "lof_score_raw"] = bsn1_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn1_lof_scores)

df_out_bsn1 = (
    pd.DataFrame({
        "customer_id": bsn1_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn1_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn1 = df_out_bsn1[["customer_id", "outlier_score_01"]]

df_out_bsn1.to_csv("bsn1_LOF_k35_scores.csv", index=False,
              float_format="%.10f")

top_ids = bsn1_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn1 = bsn1_scored.loc[top_ids].copy()
top_anomalies_bsn1.to_csv("bsn1_LOF_top.csv", index=False)

pd.read_csv("bsn1_LOF_top.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 1 LOOP
# --------------------------------------------------------------
k1 = list(range(10, 46, 2))

TOP_N_k1 = 1194

ids_to_track = [
  "SYNID0200031873",
"SYNID0200511598",
"SYNID0200285433",
"SYNID0200467375",
"SYNID0200473490",
"SYNID0200026303",
"SYNID0200164931",
"SYNID0200292620",
"SYNID0200366064",
"SYNID0200447884",
"SYNID0200755574" # LABELED
]

rank_by_k = {}

for k in k1:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn1_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=bsn1_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn1 table that has customer_id as a column
    bsn1_scored_k = bsn1_vAF_copy.copy()
    bsn1_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k1).index
    top_anomalies_k = bsn1_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"bsn1_LOF_top{TOP_N_k1}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn1_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(30, 0)
plt.title("LOF (BSN1 n = 1194): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()



# %%
# --------------------------------------------------------------
# CLUSTER 0 FIRST FIT
# --------------------------------------------------------------
# 4317 rows
bsn0_lof_model = LocalOutlierFactor(
    n_neighbors=66,
    contamination="auto",
    n_jobs=-1
    )

bsn0_lof_fitted = bsn0_lof_model.fit_predict(bsn0_vAF)

bsn0_lof_scores = pd.Series(
    bsn0_lof_model.negative_outlier_factor_,
    index=bsn0_vAF.index
)

bsn0_scored = bsn0_vAF_copy.copy()  # has customer_id column and is indexed by customer_id
bsn0_scored.loc[bsn0_lof_scores.index, "lof_score_raw"] = bsn0_lof_scores

lof_all_norm = normalize_to_unit_interval(bsn0_lof_scores)

df_out_bsn0 = (
    pd.DataFrame({
        "customer_id": bsn0_lof_scores.index,
        "outlier_score_01": lof_all_norm.values,
        "lof_score_raw": bsn0_lof_scores.values,
    })
    .sort_values(["outlier_score_01", "lof_score_raw"], ascending=[False, True])
)

df_out_bsn0 = df_out_bsn0[["customer_id", "outlier_score_01"]]

df_out_bsn0.to_csv("bsn0_LOF_k66_scores.csv", index=False,
              float_format=
              "%.10f")

top_ids = bsn0_lof_scores.sort_values(ascending=True).index
top_anomalies_bsn0 = bsn0_scored.loc[top_ids].copy()
top_anomalies_bsn0.to_csv("bsn0_LOF_top.csv", index=False)

pd.read_csv("bsn0_LOF_top.csv").head()

# %%
# --------------------------------------------------------------
# CLUSTER 0 LOOP
# --------------------------------------------------------------
k0 = list(range(40, 120, 5))

TOP_N_k0 = 200

ids_to_track = [
 "SYNID0200103724",
"SYNID0200115422",
"SYNID0200534136",
"SYNID0200974352",
"SYNID0200056757",
"SYNID0200056480",
"SYNID0200354812",
"SYNID0200353380",
"SYNID0200784709",
"SYNID0200722040"
]

rank_by_k = {}

for k in k0:
    lof_loop = LocalOutlierFactor(
        n_neighbors=k,
        contamination="auto",
        n_jobs=-1
    )

    _ = lof_loop.fit_predict(bsn0_vAF)

    lof_scores = pd.Series(
    lof_loop.negative_outlier_factor_,
    index=bsn0_vAF.index,
    name="lof_score_raw"
)

    # ranks for movement plot (1 = most outlier; most negative LOF = strongest outlier)
    rank_by_k[k] = lof_scores.rank(method="min", ascending=True).astype(int)

    # attach scores to a copy of the bsn0 table that has customer_id as a column
    bsn0_scored_k = bsn0_vAF_copy.copy()
    bsn0_scored_k.loc[lof_scores.index, "lof_score_raw"] = lof_scores

    # export top anomalies for this k
    top_ids = lof_scores.nsmallest(TOP_N_k0).index
    top_anomalies_k = bsn0_scored_k.loc[top_ids].copy()
    top_anomalies_k.to_csv(
    os.path.join(OUTPUT_DIR, f"bsn0_LOF_top{TOP_N_k0}_k{k}.csv"),
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
    os.path.join(OUTPUT_DIR, "tracked_rank_movement_bsn0_LOF.csv"),
    index=False
)

plt.figure()
for cid, g in tracked.groupby("customer_id"):
    plt.plot(g["k"], g["rank"], marker="o", markersize=2, label=str(cid))
plt.gca().invert_yaxis()
plt.xlabel("n_neighbors (k)")
plt.ylabel("Rank (1 = most outlier)")
#plt.ylim(50, 0)
plt.title("LOF (bsn0): rank movement across k")
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()


# %%
# --------------------------------------------------------------
# SCORES ONLY FOR RULE BASED ALGO
# --------------------------------------------------------------


# Combine all raw LOF scores across clusters into one Series
all_lof_scores = pd.concat([
    bsn0_lof_scores,
    bsn1_lof_scores,
    bsn2_lof_scores,
    bsn3_lof_scores,
    bsn4_lof_scores,
    bsn5_lof_scores,
    bsn6_lof_scores,
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

top_anomalies_bsn0 = compute_global_extremeness(top_anomalies_bsn0, bsn0_lof_scores)
top_anomalies_bsn1 = compute_global_extremeness(top_anomalies_bsn1, bsn1_lof_scores)
top_anomalies_bsn2 = compute_global_extremeness(top_anomalies_bsn2, bsn2_lof_scores)
top_anomalies_bsn3 = compute_global_extremeness(top_anomalies_bsn3, bsn3_lof_scores)
top_anomalies_bsn4 = compute_global_extremeness(top_anomalies_bsn4, bsn4_lof_scores)
top_anomalies_bsn5 = compute_global_extremeness(top_anomalies_bsn5, bsn5_lof_scores)
top_anomalies_bsn6 = compute_global_extremeness(top_anomalies_bsn6, bsn6_lof_scores)

# %%
bsnVA_SCORES = pd.concat([
    top_anomalies_bsn0[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_bsn1[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_bsn2[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_bsn3[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_bsn4[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_bsn5[["customer_id", "within_cluster_extremeness"]],
    top_anomalies_bsn6[["customer_id", "within_cluster_extremeness"]],
], axis=0, ignore_index=True)

bsnVA_SCORES.to_csv("bsnVA_SCORES.csv", index=False)

# %%
# Combine with individuals scores
indVA_SCORES = pd.read_csv("indVA_SCORES.csv")

clustered_LOF_scores = pd.concat([indVA_SCORES, bsnVA_SCORES], axis=0, ignore_index=True)

clustered_LOF_scores.to_csv("clustered_LOF_scores.csv", index=False)
clustered_LOF_scores.head()
# %%
import matplotlib.pyplot as plt

plt.figure()
plt.hist(all_lof_scores, bins=50, edgecolor="black")
plt.xlabel("within_cluster_extremeness")
plt.ylabel("Count")
plt.title("Distribution of Within-Cluster LOF Scores")
plt.grid(True)
plt.show()
# %%
plt.figure()
plt.hist(all_lof_scores, bins=50, edgecolor="black")
plt.xlabel("LOF Score (raw negative)")
plt.ylabel("Count")
plt.title("Distribution of Raw LOF Scores")
plt.grid(True)
plt.show()
# %%
