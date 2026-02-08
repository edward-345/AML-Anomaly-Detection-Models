# ---------------------------------------------------------------------------
# CLEANING AND DATA PROCESSING ----------------------------------------------
# ---------------------------------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.options.display.float_format = '{:.2f}'.format

def cast_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Converts known categorical columns into correct type"""
    cat_cols = [
        "customer_id",
        "transaction_id",
        "label",
        "country",
        "province",
        "city",
        "gender",
        "marital_status",
        "occupation_code",
        "industry_code",
        "merchant_category",
        "ecommerce_ind",
        "debit_credit",
        "cash_indicator"
    ]

    cols_to_cast = df.columns.intersection(cat_cols)
    df[cols_to_cast] = df[cols_to_cast].astype("category")

    return df


def build_debit_credit_features(
    df: pd.DataFrame,
    prefix: str,
    eps: float = 1.0,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Builds magnitude-, direction-, and proportion-based features
    from paired debit/credit transaction columns.
    """

    d_txn = f"{prefix}_debit_transactions"
    c_txn = f"{prefix}_credit_transactions"
    d_amt = f"{prefix}_debit_sum"
    c_amt = f"{prefix}_credit_sum"

    required = {d_txn, c_txn, d_amt, c_amt}
    if not required.issubset(df.columns):
        # silently skip if this channel doesn't exist
        return df

    # ---- totals (activity intensity)
    df[f"{prefix}_total_transactions"] = df[d_txn] + df[c_txn]
    df[f"{prefix}_total_amount"] = df[d_amt] + df[c_amt]

    # ---- net flow (directionality)
    df[f"{prefix}_net_transactions"] = df[d_txn] - df[c_txn]
    df[f"{prefix}_net_amount"] = df[d_amt] - df[c_amt]

    # ---- ratios (smoothed)
    df[f"{prefix}_txn_ratio"] = df[d_txn] / (df[c_txn] + eps)
    df[f"{prefix}_amt_ratio"] = df[d_amt] / (df[c_amt] + eps)

    # ---- proportions (bounded, LOF-friendly)
    df[f"{prefix}_debit_txn_share"] = (
        df[d_txn] / (df[d_txn] + df[c_txn] + eps)
    )
    df[f"{prefix}_debit_amt_share"] = (
        df[d_amt] / (df[d_amt] + df[c_amt] + eps)
    )

    # ---- log-asymmetry (stable under PCA / distance)
    df[f"{prefix}_log_txn_ratio"] = (
        np.log1p(df[d_txn]) - np.log1p(df[c_txn])
    )
    df[f"{prefix}_log_amt_ratio"] = (
        np.log1p(df[d_amt]) - np.log1p(df[c_amt])
    )

    if drop_original:
        df.drop(columns=[d_txn, c_txn, d_amt, c_amt], inplace=True)

    return df