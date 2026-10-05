#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -m millstlabs.next_experiment "$@"
