# Supervised Tension Detection (Production‑ready Baseline)

This module provides a practical, explainable baseline for 3‑class (green/yellow/red) tension classification over conversational turns. It includes:

- Data loader for CSV/JSONL with multi‑rater fields
- TF‑IDF + numeric features (latency_ms, token_count, previous_tension_state)
- Multinomial Logistic Regression with stratified CV grid search
- Joblib serialization of model, vectorizer, scaler, label encoder
- CLI for training and batch/interactive inference

## Install Minimal Dependencies

- Create/activate a venv, then:
  - `pip install -r requirements-tension.txt`

## Dataset Expectations

Each record must include at least:
- `text` (string)
- `tension_label` or `consensus_label` in {green,yellow,red}

Optional numeric features:
- `latency_ms` (float), `token_count` (int; auto‑computed if missing), `previous_tension_state` (int in {0,1,2}; default 1)

Supported inputs: CSV or JSON Lines (JSONL). For JSONL, each line is a JSON object.

## Train

- `python3 vesselos-dev-research/ml/tension/train.py --data path/to/data.jsonl --label consensus_label --out artifacts/tension`  
  Options: `--test-size 0.15 --cv 5 --seed 42`.

Artifacts written under `--out/<timestamp>/`:
- `tension_classifier.joblib`, `tfidf_vectorizer.joblib`, `numeric_scaler.joblib`, `label_encoder.joblib`
- `metrics.json`, `classification_report.json`, `confusion_matrix.png`

## Predict

- Single example:  
  `python3 vesselos-dev-research/ml/tension/infer.py --model-dir artifacts/tension/<ts> --text "I disagree with that." --latency-ms 1200 --previous 1`

- Batch CSV/JSONL:  
  `python3 vesselos-dev-research/ml/tension/infer.py --model-dir artifacts/tension/<ts> --in data.jsonl --out preds.jsonl`

## Notes

- See docs/practical-tension-classification-production-guide.md for end‑to‑end methodology.
- Start with full‑agreement examples; then add majority cases; use active learning for ambiguous cases.

