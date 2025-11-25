#!/usr/bin/env bash
# Orchestrate Tink export sorting: rename → group → docs+OCR → indexes
# Optional: ingest summaries into Kira Prime (Limnus) and reindex.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

TINK_REPO_DEFAULT="${ROOT}/tink-full-export-repo"
TINK_REPO="${TINK_REPO_DEFAULT}"
DO_RENAME=1
DO_GROUP=1
DO_DOCS=1
DO_INDEX=1
DO_OCR=1
DO_KIRA=0
KIRA_WORKSPACE="tink"
RENAME_LIMIT=0
RENAME_MODEL="nlpconnect/vit-gpt2-image-captioning"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --repo <path>           Path to tink-full-export-repo (default: ${TINK_REPO_DEFAULT})
  --skip-rename           Skip content-based renaming
  --skip-group            Skip grouping by theme
  --skip-docs             Skip companion docs generation
  --skip-index            Skip HTML index/browser builds
  --no-ocr                Disable OCR in companion generation
  --limit <N>             Limit images for rename step (0 = all)
  --model <id>            HF caption model id for rename (default: ${RENAME_MODEL})
  --kira                  Ingest chat summaries into Kira Prime Limnus and reindex
  --kira-workspace <id>   Workspace id for Kira ingestion (default: tink)
  -h, --help              Show this help

This script wires together existing repo tools to sort Tink's export.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) TINK_REPO="$2"; shift 2 ;;
    --skip-rename) DO_RENAME=0; shift ;;
    --skip-group) DO_GROUP=0; shift ;;
    --skip-docs) DO_DOCS=0; shift ;;
    --skip-index) DO_INDEX=0; shift ;;
    --no-ocr) DO_OCR=0; shift ;;
    --limit) RENAME_LIMIT="$2"; shift 2 ;;
    --model) RENAME_MODEL="$2"; shift 2 ;;
    --kira) DO_KIRA=1; shift ;;
    --kira-workspace) KIRA_WORKSPACE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d "${TINK_REPO}" ]]; then
  echo "ERROR: repo directory not found: ${TINK_REPO}" >&2
  exit 2
fi

IMG_DIR="${TINK_REPO}/Tink Full Export/images"
META_DIR="${TINK_REPO}/Tink Full Export/metadata"
SUMM_DIR="${TINK_REPO}/Tink Full Export/chat_summaries"

if [[ ! -d "${IMG_DIR}" ]]; then
  echo "ERROR: images directory not found: ${IMG_DIR}" >&2
  exit 2
fi

mkdir -p "${META_DIR}"

echo "Repo: ${TINK_REPO}"
echo "Images: ${IMG_DIR}"

# ------------------------------------------------------------- rename by content
if [[ ${DO_RENAME} -eq 1 ]]; then
  echo "[1/5] Rename by visual content (may download model weights)…"
  PY_OK=1
  python3 - <<'PY'
try:
  import PIL  # noqa: F401
  import transformers  # noqa: F401
  import torch  # noqa: F401
except Exception as e:
  raise SystemExit(1)
PY
  if [[ $? -ne 0 ]]; then
    echo "WARN: Missing deps for rename (transformers, torch, pillow). Skipping rename step." >&2
    DO_RENAME=0
  fi
fi

if [[ ${DO_RENAME} -eq 1 ]]; then
  set +e
  rename_cmd=(python3 "${ROOT}/scripts/rename_images_by_content.py"
              --images-dir "${IMG_DIR}"
              --out "${META_DIR}/rename_manifest.csv"
              --model "${RENAME_MODEL}"
              --apply)
  if [[ ${RENAME_LIMIT} -gt 0 ]]; then
    rename_cmd+=(--limit "${RENAME_LIMIT}")
  fi
  "${rename_cmd[@]}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "WARN: rename step failed with code ${rc}; continuing." >&2
  fi
fi

# ---------------------------------------------------------------- group by theme
if [[ ${DO_GROUP} -eq 1 ]]; then
  echo "[2/5] Group images by theme…"
  python3 "${ROOT}/scripts/group_images_by_theme.py" \
    --images-dir "${IMG_DIR}" \
    --rename-manifest "${META_DIR}/rename_manifest.csv" \
    --out "${META_DIR}/grouping_manifest.csv" \
    --apply || echo "WARN: grouping step reported an issue; continuing." >&2
fi

# ------------------------------------------------------ companions (+OCR) + index
if [[ ${DO_DOCS} -eq 1 ]]; then
  echo "[3/5] Generate companion docs (OCR=${DO_OCR})…"
  docs_cmd=(python3 "${ROOT}/scripts/generate_image_companions.py")
  if [[ ${DO_OCR} -eq 1 ]]; then
    docs_cmd+=(--ocr)
  fi
  "${docs_cmd[@]}"
fi

if [[ ${DO_INDEX} -eq 1 ]]; then
  echo "[4/5] Rebuild gallery index and conversations browser…"
  python3 "${ROOT}/scripts/build_html_index.py"
  python3 "${ROOT}/scripts/extract_chat_summaries.py"
  python3 "${ROOT}/scripts/build_chat_summary_html.py"
  python3 "${ROOT}/scripts/build_chat_summaries_index_html.py"
fi

# -------------------------------------------------------------- Kira ingestion
if [[ ${DO_KIRA} -eq 1 ]]; then
  echo "[5/5] Kira Prime: ingest summaries into Limnus (workspace=${KIRA_WORKSPACE})…"
  # Ensure basic deps for Kira (best-effort; requirements.txt should cover)
  if [[ -f "${ROOT}/kira-prime/requirements.txt" ]]; then
    pip3 install -r "${ROOT}/kira-prime/requirements.txt" >/dev/null 2>&1 || true
  fi
  # Ingest first 200 lines of each summary MD
  shopt -s nullglob
  for md in "${SUMM_DIR}"/*.md; do
    text="$(sed '1,200q' "${md}")"
    (cd "${ROOT}/kira-prime" && python3 vesselos.py limnus cache "${text}") >/dev/null 2>&1 || true
  done
  # Build vector index
  (cd "${ROOT}/kira-prime" && python3 vesselos.py limnus reindex --backend sbert) || true
  echo "Kira ingestion complete. Try: (cd kira-prime && python3 vesselos.py limnus recall \"hyperfollow integration plan\")"
fi

echo "Done. Outputs:"
echo "- Gallery index: ${TINK_REPO}/index.html"
echo "- Theme pages:   ${TINK_REPO}/gallery/*.html"
echo "- Image docs:    ${TINK_REPO}/Tink Full Export/image_docs/"
echo "- Summaries:     ${TINK_REPO}/Tink Full Export/chat_summaries{,_html}/"

