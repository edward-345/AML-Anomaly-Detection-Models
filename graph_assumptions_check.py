# %%
"""
STEP 0 — ASSUMPTION CHECKS FOR GRAPHICAL MODELING (AML)

Assumptions tested:
1. Structural heterogeneity
2. Behavioral similarity (non-i.i.d.)
3. Locality (kNN meaningful)
4. Temporal stability (optional)
5. Scaling robustness (RobustScaler)

Uses only net transaction features.
No labels required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# -------------------------------
# 0. LOAD + PREPROCESS DATA
# -------------------------------

from data_preprocessing import cast_categorical
from data_preprocessing import build_debit_credit_features

pd.set_option("display.max_columns", None)

bsn = pd.read_csv(
    "/Users/dangernoodle_/Desktop/DATA/DataTables/clean_businesses.csv"
)

# type casting
cast_categorical(bsn)
bsn["established_date"] = pd.to_datetime(bsn["established_date"])
bsn["onboard_date"] = pd.to_datetime(bsn["onboard_date"])

# -------------------------------
# Feature engineering
# -------------------------------

channels = ["wire", "eft", "emt", "west", "cheque", "card", "abm"]
bsn_new = bsn.copy()

for ch in channels:
    bsn_new = build_debit_credit_features(bsn_new, ch)

bsn_new = bsn_new.set_index("customer_id")

# -------------------------------
# Select net features ONLY
# -------------------------------

net_cols = [
    "wire_net_transactions",
    "eft_net_transactions",
    "emt_net_transactions",
    "west_net_transactions",
    "cheque_net_transactions",
    "card_net_transactions",
    "abm_net_transactions",
    "wire_net_amount",
    "eft_net_amount",
    "emt_net_amount",
    "west_net_amount",
    "cheque_net_amount",
    "card_net_amount",
    "abm_net_amount",
]

bsn_net = bsn_new[net_cols].copy()

# -------------------------------
# Drop all-zero accounts (CRITICAL)
# -------------------------------

nonzero_mask = (bsn_net.sum(axis=1) != 0)
bsn_net = bsn_net.loc[nonzero_mask]

# -------------------------------
# 1. FEATURE MATRIX
# -------------------------------

X_raw = bsn_net.fillna(0)

# -------------------------------
# 2. ROBUST SCALING
# -------------------------------

scaler = RobustScaler()
X = scaler.fit_transform(X_raw)

# -------------------------------
# ASSUMPTION 1
# Structural heterogeneity
# -------------------------------

cov = np.cov(X, rowvar=False)
eigs = np.linalg.eigvalsh(cov)

plt.figure()
plt.plot(np.sort(eigs)[::-1])
plt.title("Eigenvalue Spectrum (Structural Heterogeneity)")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.show()

print("Top 5 eigenvalues:", np.sort(eigs)[-5:])
print("Near-zero eigenvalues:", np.sum(eigs < 1e-6))

# Interpretation:
# Few dominant eigenvalues → heterogeneous behavior exists

# -------------------------------
# ASSUMPTION 2
# Behavioral similarity (non-i.i.d.)
# -------------------------------

sim = cosine_similarity(X)

rng = np.random.default_rng(42)
n = X.shape[0]

random_pairs = [
    sim[rng.integers(0, n), rng.integers(0, n)]
    for _ in range(1000)
]

knn = NearestNeighbors(n_neighbors=6, metric="cosine")
knn.fit(X)
distances, indices = knn.kneighbors(X)

neighbor_sims = []
for i in range(n):
    for j in indices[i][1:]:
        neighbor_sims.append(sim[i, j])

print("Mean random similarity:", np.mean(random_pairs))
print("Mean neighbor similarity:", np.mean(neighbor_sims))

plt.figure()
plt.hist(random_pairs, bins=30, alpha=0.5, label="Random pairs")
plt.hist(neighbor_sims, bins=30, alpha=0.5, label="kNN neighbors")
plt.legend()
plt.title("Similarity: Random vs Local Neighbors")
plt.show()

# Interpretation:
# Neighbor similarity >> random similarity → relational dependence

# -------------------------------
# ASSUMPTION 3
# Locality is meaningful
# -------------------------------

plt.figure()
plt.plot(np.sort(distances[:, -1]))
plt.title("Distance to k-th Nearest Neighbor (Cosine Distance)")
plt.xlabel("Accounts (sorted)")
plt.ylabel("Distance")
plt.show()

# Interpretation:
# Smooth curve → stable local neighborhoods

# -------------------------------
# ASSUMPTION 4
# Scaling robustness sanity check
# -------------------------------

pca_raw = PCA(n_components=2).fit_transform(X_raw)
pca_scaled = PCA(n_components=2).fit_transform(X)

plt.figure()
plt.scatter(pca_raw[:, 0], pca_raw[:, 1], alpha=0.3)
plt.title("PCA before scaling")
plt.show()

plt.figure()
plt.scatter(pca_scaled[:, 0], pca_scaled[:, 1], alpha=0.3)
plt.title("PCA after RobustScaler")
plt.show()

# -------------------------------
print("\nSTEP 0 COMPLETE")
print("If Assumptions 1–3 hold visually and numerically,")
print("you are justified in proceeding to graph construction.")

# %%
# ============================================================
# STEP 1 — DEFINE NODES + NODE FEATURES (BUSINESS ONLY, STATIC)
# Uses objects already created in Step 0:
#   - bsn_new (indexed by customer_id)
#   - bsn_net (net_cols filtered + nonzero accounts)
#   - net_cols
# ============================================================

# 1) Define the node set (business accounts with nonzero net activity)
node_ids = bsn_net.index.to_numpy()   # already filtered by nonzero_mask
print(f"\nSTEP 1: Nodes defined (active business accounts): {len(node_ids)}")

# 2) Assemble node feature table (net_* only for now)
node_df = bsn_new.loc[node_ids, net_cols].copy().fillna(0)

# Safety checks: alignment + no duplicates
assert node_df.index.is_unique, "Duplicate customer_id in node_df index."
assert node_df.shape[0] == len(node_ids), "Mismatch between node_ids and node_df rows."

# 3) Optional: add KYC numeric features (uncomment if you want them as node features)
# kyc_num_cols = ["employee_count", "sales", "business_age", "account_age"]
# kyc_num_cols = [c for c in kyc_num_cols if c in bsn_new.columns]
# if kyc_num_cols:
#     node_df = pd.concat([node_df, bsn_new.loc[node_ids, kyc_num_cols].fillna(0)], axis=1)

# 4) Create a stable node index mapping (needed later for PyG/DGL edge_index)
node_df = node_df.sort_index()  # deterministic ordering
node_to_idx = {cid: i for i, cid in enumerate(node_df.index)}
idx_to_node = node_df.index.to_numpy()

print("STEP 1: node_df shape:", node_df.shape)
print("STEP 1: Example node ids:", idx_to_node[:5])

# 5) Save artifacts for Step 2+ (recommended)
node_df.to_parquet("node_features_business.parquet")
pd.Series(node_to_idx).to_csv("node_to_idx_business.csv", header=False)

print("STEP 1 COMPLETE")
print("Saved: node_features_business.parquet and node_to_idx_business.csv")

# %%
np.save("X_raw.npy", node_df.values)

# ============================================================
# STEP 2 — BUILD kNN SIMILARITY GRAPH
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

print("\nSTEP 2 — Building similarity graph")

# -------------------------------
# Load node features
# -------------------------------

X_raw = np.load("X_raw.npy")

# IMPORTANT: scale before similarity (same as Step 0)
scaler = RobustScaler()
X = scaler.fit_transform(X_raw)

N = X.shape[0]

# -------------------------------
# kNN construction
# -------------------------------

k = 8   # <<< adjust if desired

knn = NearestNeighbors(n_neighbors=k+1, metric="cosine")
knn.fit(X)

distances, indices = knn.kneighbors(X)

# cosine distance = 1 - similarity
similarities = 1 - distances

# -------------------------------
# Build edge list
# -------------------------------

rows = []
cols = []
weights = []

for i in range(N):
    for j_idx, sim in zip(indices[i][1:], similarities[i][1:]):  # skip self
        rows.append(i)
        cols.append(j_idx)
        weights.append(sim)

# Make undirected (symmetrize) — SAFE
rows = np.array(rows)
cols = np.array(cols)
weights = np.array(weights)

rows_sym = np.concatenate([rows, cols])
cols_sym = np.concatenate([cols, rows])
weights_sym = np.concatenate([weights, weights])

edge_index = np.vstack([rows_sym, cols_sym])
edge_weight = weights_sym
# -------------------------------
# Save outputs
# -------------------------------

np.save("edge_index.npy", edge_index)
np.save("edge_weight.npy", edge_weight)

print(f"Nodes: {N}")
print(f"Edges (directed, after sym): {edge_index.shape[1]}")
print("Saved edge_index.npy and edge_weight.npy")

# -------------------------------
# Quick sanity stats
# -------------------------------

deg = np.bincount(edge_index[0], minlength=N)
print("Mean degree:", deg.mean())
print("Min degree:", deg.min())
print("Max degree:", deg.max())

# %%
# --- Sanity checks (run once) ---
N = X.shape[0]  # same X you used for kNN
assert edge_index.shape[1] == edge_weight.shape[0], "edge_index/edge_weight length mismatch"
assert edge_index.min() >= 0 and edge_index.max() < N, "edge_index out of bounds"

deg = np.bincount(edge_index[0], minlength=N)
print("Nodes:", N)
print("Edges (directed, after sym):", edge_index.shape[1])
print("Mean degree:", deg.mean(), "Min degree:", deg.min(), "Max degree:", deg.max())
print("Any isolates?", np.sum(deg == 0))
print("Edge weight range:", float(edge_weight.min()), "to", float(edge_weight.max()))

# %%
