import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def cast_binary_and_categorical(
    df: pd.DataFrame,
    *,
    binary_as_category: bool = True,
    object_as_category: bool = True,
    max_unique_for_int_category: int = 50,
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
