from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt

from .features import FeatureBundle, fit_transform_features, transform_features, TEXT_COL


LABELS = ["green", "yellow", "red"]


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    # JSONL default
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    if label_col not in df.columns:
        # fallback to tension_label
        if "tension_label" not in df.columns:
            raise ValueError(f"label column '{label_col}' not found and no 'tension_label' present")
        label_col = "tension_label"
    y_raw = df[label_col].astype(str)
    le = LabelEncoder()
    # Preserve ordering green<yellow<red when present
    try:
        le.fit(LABELS)
        # map unknowns via fit on combined labels
        le.fit(list(dict.fromkeys(LABELS + sorted(y_raw.unique().tolist()))))
    except Exception:
        le.fit(y_raw)
    y = le.transform(y_raw)
    X_df = df[[TEXT_COL, "latency_ms", "token_count", "previous_tension_state"]].copy()
    return X_df, y, le


def plot_confusion(cm: np.ndarray, labels: List[str], out: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train supervised tension classifier (LR + TFIDF + numeric)")
    ap.add_argument("--data", type=Path, required=True, help="Path to CSV or JSONL dataset")
    ap.add_argument("--label", default="consensus_label", help="Label column (default: consensus_label)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/tension"), help="Output root directory")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args(argv)

    df = load_any(ns.data)
    if TEXT_COL not in df.columns:
        raise SystemExit(f"missing required column '{TEXT_COL}'")

    X_df, y, le = split_xy(df, ns.label)

    # train/val/test: first split test, then val from train
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_df, y, test_size=ns.test_size, stratify=y, random_state=ns.seed
    )
    val_size = ns.test_size  # keep same proportion for val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1.0 - ns.test_size), stratify=y_temp, random_state=ns.seed
    )

    # Fit features on train
    X_train_mat, fb = fit_transform_features(X_train)
    X_val_mat = transform_features(X_val, fb)
    X_test_mat = transform_features(X_test, fb)

    # Grid search LR
    base = LogisticRegression(
        solver="lbfgs",
        multi_class="multinomial",
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=ns.seed,
    )
    grid = GridSearchCV(
        base,
        param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=StratifiedKFold(ns.cv, shuffle=True, random_state=ns.seed),
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_mat, y_train)
    clf = grid.best_estimator_

    # Evaluate
    def rep(X, y_true) -> Dict:
        y_pred = clf.predict(X)
        return json.loads(
            classification_report(y_true, y_pred, target_names=list(le.inverse_transform([0, 1, 2])), digits=3, output_dict=True)
        )

    report_val = rep(X_val_mat, y_val)
    y_pred_test = clf.predict(X_test_mat)
    report_test = json.loads(
        classification_report(y_test, y_pred_test, target_names=list(le.inverse_transform([0, 1, 2])), digits=3, output_dict=True)
    )
    weighted_f1 = f1_score(y_test, y_pred_test, average="weighted")
    macro_f1 = f1_score(y_test, y_pred_test, average="macro")
    cm = confusion_matrix(y_test, y_pred_test)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ns.out / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    dump(clf, out_dir / "tension_classifier.joblib", compress=3)
    dump(fb.vectorizer, out_dir / "tfidf_vectorizer.joblib", compress=3)
    dump(fb.scaler, out_dir / "numeric_scaler.joblib", compress=3)
    dump(le, out_dir / "label_encoder.joblib", compress=3)

    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump({"val": report_val, "test": report_test}, f, indent=2)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_C": getattr(clf, "C", None),
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "classes": list(le.classes_.astype(str)),
                "n_train": int(X_train_mat.shape[0]),
                "n_val": int(X_val_mat.shape[0]),
                "n_test": int(X_test_mat.shape[0]),
            },
            f,
            indent=2,
        )

    plot_confusion(cm, list(le.inverse_transform([0, 1, 2])), out_dir / "confusion_matrix.png")

    print(f"Artifacts written to {out_dir}")
    print(json.dumps({"weighted_f1": weighted_f1, "macro_f1": macro_f1, "best_C": getattr(clf, 'C', None)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

