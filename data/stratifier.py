"""
data/stratifier.py
==================
Stratified split for PlantDiagBench-style data (Appendix C).

"The test subset is stratified by the configured stratify columns (e.g. crop×severity).
 The calibration split is drawn from remaining images not in the test set,
 stratified similarly to limit distribution shift."
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    n_test: int = 5000,
    n_cal: int = 500,
    stratify_cols: List[str] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_test, df_cal).

    - df_test  : n_test images, stratified by stratify_cols
    - df_cal   : n_cal  images from *remaining* rows, stratified by stratify_cols
                 (disjoint from test — Appendix C)

    If the dataframe has fewer than n_test + n_cal rows, we use as many as available.
    """
    if stratify_cols is None:
        stratify_cols = []

    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Build a combined stratification key
    available_cols = [c for c in stratify_cols if c in df.columns]
    if available_cols:
        strat_key = df[available_cols].astype(str).agg("__".join, axis=1)
    else:
        strat_key = None

    total_needed = n_test + n_cal
    if len(df) < total_needed:
        # Use all available; split proportionally
        n_test = min(n_test, int(len(df) * (n_test / total_needed)))
        n_cal = len(df) - n_test

    # First: carve out n_test rows
    if strat_key is not None and strat_key.nunique() > 1:
        try:
            idx_test, idx_rest = train_test_split(
                df.index,
                train_size=n_test,
                stratify=strat_key,
                random_state=seed,
            )
        except ValueError:
            # Fallback if stratification fails (very small groups)
            idx_test = rng.choice(df.index, size=n_test, replace=False)
            idx_rest = df.index.difference(idx_test)
    else:
        idx_test = rng.choice(df.index, size=n_test, replace=False)
        idx_rest = df.index.difference(idx_test)

    df_test = df.loc[idx_test].reset_index(drop=True)
    df_remaining = df.loc[idx_rest].reset_index(drop=True)

    # Second: carve out n_cal from remaining
    if len(df_remaining) == 0:
        return df_test, pd.DataFrame(columns=df.columns)

    n_cal = min(n_cal, len(df_remaining))
    if strat_key is not None and strat_key.nunique() > 1:
        strat_rest = strat_key.loc[idx_rest].reset_index(drop=True)
        try:
            idx_cal, _ = train_test_split(
                df_remaining.index,
                train_size=n_cal,
                stratify=strat_rest,
                random_state=seed,
            )
        except ValueError:
            idx_cal = rng.choice(df_remaining.index, size=n_cal, replace=False)
    else:
        idx_cal = rng.choice(df_remaining.index, size=n_cal, replace=False)

    df_cal = df_remaining.loc[idx_cal].reset_index(drop=True)
    return df_test, df_cal
