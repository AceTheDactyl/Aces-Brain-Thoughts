from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load
from scipy.sparse import hstack, csr_matrix

from .features import FeatureBundle, transform_features, TEXT_COL, ensure_numeric_features


class TensionDetector:
    def __init__(self, model_dir: Path | str = "artifacts/tension/latest") -> None:
        p = Path(model_dir)
        self.clf = load(p / "tension_classifier.joblib")
        self.vectorizer = load(p / "tfidf_vectorizer.joblib")
        self.scaler = load(p / "numeric_scaler.joblib")
        self.label_encoder = load(p / "label_encoder.joblib")

    def _features(self, df: pd.DataFrame) -> csr_matrix:
        fb = FeatureBundle(vectorizer=self.vectorizer, scaler=self.scaler)
        return transform_features(df, fb)

    def predict_one(self, text: str, latency_ms: float, token_count: int | None = None, previous_state: int = 1) -> Dict:
        if token_count is None:
            token_count = len((text or "").split())
        df = pd.DataFrame(
            [
                {
                    TEXT_COL: text,
                    "latency_ms": float(latency_ms),
                    "token_count": int(token_count),
                    "previous_tension_state": int(previous_state),
                }
            ]
        )
        X = self._features(df)
        pred_idx = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X)[0]
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        prob_dict = {self.label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
        return {"label": str(label), "proba": prob_dict}

    def predict_file(self, in_path: Path, out_path: Path | None = None) -> List[Dict]:
        df = self._load_any(in_path)
        df = ensure_numeric_features(df)
        X = self._features(df)
        pred_idx = self.clf.predict(X)
        proba = self.clf.predict_proba(X)
        labels = self.label_encoder.inverse_transform(pred_idx)
        out_rows: List[Dict] = []
        for i in range(len(df)):
            prob_dict = {self.label_encoder.inverse_transform([j])[0]: float(proba[i, j]) for j in range(proba.shape[1])}
            out_rows.append({"label": str(labels[i]), "proba": prob_dict})
        if out_path:
            with out_path.open("w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row) + "\n")
        return out_rows

    @staticmethod
    def _load_any(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inference CLI for tension detector")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--token-count", type=int, default=None)
    ap.add_argument("--previous", type=int, default=1)
    ap.add_argument("--in", dest="in_path", type=Path, default=None)
    ap.add_argument("--out", dest="out_path", type=Path, default=None)
    ns = ap.parse_args(argv)

    det = TensionDetector(ns.model_dir)
    if ns.in_path is not None:
        det.predict_file(ns.in_path, ns.out_path)
        return 0
    if ns.text is None:
        raise SystemExit("Provide --text for single example or --in for batch input")
    res = det.predict_one(ns.text, ns.latency_ms, ns.token_count, ns.previous)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

