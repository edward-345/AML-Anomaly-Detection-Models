# %%
import pandas as pd
import numpy as np

# %%
wire = pd.read_csv("clean_txn_wire.csv")

print(wire.head())

# %%
print(wire.dtypes)

# %%
wire["customer_id"] = wire["customer_id"].astype("category")
wire["customer_id"] = wire["customer_id"].astype("category")
wire["amount_cad"] = wire["amount_cad"].astype("float32")
wire["debit_credit"] = wire["debit_credit"].astype("category")
wire["transaction_datetime"] = pd.to_datetime(wire["transaction_datetime"])

print(wire.dtypes)

# %%
print(wire.head())

# %%
print(wire.tail())
# %%
print(wire.shape)

# %%
print(wire.columns)
# %%
print(wire.info())

# %%
wire.describe().style.format("{:.2f}")


# %%
import sklearn as sk
# %%
from sklearn.neighbors import LocalOutlierFactor

# %%
