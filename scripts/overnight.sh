#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv. Install with python3 -m venv .venv and .venv/bin/python -m pip install -e ".[train,notebook]"' >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m millstlabs.batch "$@"
