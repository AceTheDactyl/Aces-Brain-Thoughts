from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


TEXT_COL = "text"


@dataclass
class FeatureBundle:
    vectorizer: TfidfVectorizer
    scaler: StandardScaler


def default_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=1500,
        min_df=5,
        max_df=0.7,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )


NUMERIC_COLS = [
    "latency_ms",
    "token_count",
    "previous_tension_state",
]


def ensure_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # token_count from text if missing
    if "token_count" not in out.columns or out["token_count"].isna().any():
        out["token_count"] = out.get(TEXT_COL, pd.Series("", index=out.index)).fillna("").apply(
            lambda s: int(len(s.split()))
        )
    # default previous state = 1 (yellow/neutral)
    if "previous_tension_state" not in out.columns:
        out["previous_tension_state"] = 1
    out["previous_tension_state"] = out["previous_tension_state"].fillna(1).astype(int)
    # latency default to median if present otherwise 0
    if "latency_ms" not in out.columns:
        out["latency_ms"] = 0.0
    else:
        med = float(out["latency_ms"].dropna().median()) if out["latency_ms"].notna().any() else 0.0
        out["latency_ms"] = out["latency_ms"].fillna(med).astype(float)
    return out


def fit_transform_features(df: pd.DataFrame) -> Tuple[csr_matrix, FeatureBundle]:
    df2 = ensure_numeric_features(df)
    vec = default_vectorizer()
    X_text = vec.fit_transform(df2[TEXT_COL].fillna(""))
    scaler = StandardScaler(with_mean=False)
    X_num_dense = scaler.fit_transform(df2[NUMERIC_COLS].values.astype(float))
    # Convert numeric to sparse and hstack to preserve sparsity
    X_num = csr_matrix(X_num_dense)
    X = hstack([X_text, X_num], format="csr")
    return X, FeatureBundle(vectorizer=vec, scaler=scaler)


def transform_features(df: pd.DataFrame, fb: FeatureBundle) -> csr_matrix:
    df2 = ensure_numeric_features(df)
    X_text = fb.vectorizer.transform(df2[TEXT_COL].fillna(""))
    X_num_dense = fb.scaler.transform(df2[NUMERIC_COLS].values.astype(float))
    X_num = csr_matrix(X_num_dense)
    return hstack([X_text, X_num], format="csr")

