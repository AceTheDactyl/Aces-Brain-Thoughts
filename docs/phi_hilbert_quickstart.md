# φHILBERT-11 Quickstart (Tink Sonification)

This guide shows the end‑to‑end flow to generate post‑hoc sonifications for Tink conversations, browse them, and run a live renderer.

## 1) Environment

- Create/activate venv and install minimal deps:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-tink.txt
```

## 2) One conversation (chat.html)

```
# Generates metrics → trajectories → audio → exports (summary, timeline, prompt)
python3 tink-full-export-repo/scripts/rails_run_all.py --conv 1 --seconds 30
```

Outputs (under Tink repo):
- Tink Full Export/data/metrics_conv_0001.json
- Tink Full Export/data/trajectories_conv_0001.json
- Tink Full Export/data/sonification_conv_0001.wav
- Tink Full Export/data/export_conv_0001/(summary.md, tension_timeline.png, ai_prompt.txt)

## 3) Batch + Index

```
python3 tink-full-export-repo/scripts/rails_batch.py --limit 3 --seconds 15
python3 tink-full-export-repo/scripts/rails_build_index.py
```

Open sonifications index:
- tink-full-export-repo/Tink Full Export/phi_hilbert/index.html

## 4) Live renderer (browser)

```
# Terminal A: serve static files
python3 -m http.server

# Terminal B: stream segments over WebSocket from trajectories
python3 tink-full-export-repo/scripts/rails_live_ws.py \
    --traj "tink-full-export-repo/Tink Full Export/data/trajectories_conv_0001.json" \
    --host 127.0.0.1 --port 8765 --loop
```

Open in browser:
- http://localhost:8000/tink-full-export-repo/Tink%20Full%20Export/phi_hilbert/live.html
- Click Connect (ws://127.0.0.1:8765)

## 5) Sonify from CSV or JSON logs

CSV format (headers): `timestamp,speaker,text[,tension]`
- A template is included in both repos:
  - Monorepo: docs/phi_hilbert_session_template.csv
  - Tink repo: tink-full-export-repo/Tink Full Export/phi_hilbert_session_template.csv

Run directly:
```
python3 tink-full-export-repo/scripts/rails_run_all.py \
  --log docs/phi_hilbert_session_template.csv \
  --seconds 20
```

## 6) Kira Prime integration (post‑hoc)

```
# JSON/CSV input (attaches artifacts into TRIAD workspace outputs)
(cd kira-prime && python3 vesselos.py audit sonify \
  --workspace default --repo ../tink-full-export-repo \
  --log ../docs/phi_hilbert_session_template.csv --seconds 20)

# CSV alias
(cd kira-prime && python3 vesselos.py audit sonify-csv \
  --workspace default --repo ../tink-full-export-repo \
  --csv ../docs/phi_hilbert_session_template.csv --seconds 20)

# chat.html (by index)
(cd kira-prime && python3 vesselos.py audit sonify \
  --workspace default --repo ../tink-full-export-repo \
  --conv 1 --seconds 30)
```

Artifacts copied to:
- kira-prime/workspaces/<workspace>/outputs/sonify/<conv_tag>/

## 7) Troubleshooting
- Serve files via `python3 -m http.server` for browser fetches.
- If audio is quiet, raise the master volume slider in live.html.
- For very long sessions, increase `--seconds` or use batch mode.

