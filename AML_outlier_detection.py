# ---------------------------------------------------------------------------
# CLEANING AND DATA PROCESSING ----------------------------------------------
# ---------------------------------------------------------------------------

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.options.display.float_format = '{:.2f}'.format
# %%
def cast_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Converts known categorical columns into correct type"""
    cat_cols = [
        "customer_id",
        "transaction_id"
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

# %%
def cast_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Converts known datetime columns into correct type"""
    datetime_cols = [
        "birth_date",
        "established_date",
        "onboard_date"
    ]

    return df