#!/usr/bin/env bash
# Bootstrap a venv, install minimal deps, and run the Tink sort pipeline.
# Wraps scripts/tink_sort.sh with sane defaults.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
VENV_DIR="${ROOT}/venv"
REQ_FILE="${ROOT}/requirements-tink.txt"

# Default repo path if not provided via --repo
DEFAULT_REPO="${ROOT}/tink-full-export-repo"

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install --quiet --upgrade pip
  if [[ -f "${REQ_FILE}" ]]; then
    python -m pip install --quiet -r "${REQ_FILE}"
  fi
}

main() {
  ensure_venv
  # Always supply a default --repo first; user-provided args can override it later
  if [[ ! -x "${ROOT}/scripts/tink_sort.sh" ]]; then
    echo "ERROR: scripts/tink_sort.sh not found or not executable" >&2
    exit 2
  fi
  "${ROOT}/scripts/tink_sort.sh" --repo "${DEFAULT_REPO}" "$@"
}

main "$@"

