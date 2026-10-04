#!/usr/bin/env bash
# Portable CPU setup and notebook launcher for macOS, Linux, and Codespaces.
set -euo pipefail

notebook_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$notebook_root"
notebook_python="$notebook_root/.venv/bin/python"

if [[ ! -x "$notebook_python" ]]; then
  "${PYTHON:-python3}" -m venv "$notebook_root/.venv"
fi
"$notebook_python" -c 'import sys; sys.exit("Python 3.11+ is required; set PYTHON to a compatible executable.") if sys.version_info < (3, 11) else None'

export PATH="$notebook_root/.venv/bin:$PATH"
export HF_HOME="${HF_HOME:-$notebook_root/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$notebook_root/.cache/matplotlib}"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-$notebook_root/.cache/jupyter}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$notebook_root/.cache/jupyter-runtime}"
export IPYTHONDIR="${IPYTHONDIR:-$notebook_root/.cache/ipython}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"

if ! "$notebook_python" -c 'import torch, transformers, peft, jupyterlab, ipykernel, matplotlib, nbclient, millstlabs' >/dev/null 2>&1; then
  echo "Installing notebook dependencies into $notebook_root/.venv (first launch only)."
  # The CPU wheel avoids downloading CUDA libraries on a fresh Linux Codespace.
  if [[ "$(uname -s)" == Linux ]] && ! "$notebook_python" -c 'import torch' >/dev/null 2>&1; then
    "$notebook_python" -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
  fi
  "$notebook_python" -m pip install -e '.[train,notebook]'
fi

notebook_port="${PORT:-8888}"
notebook_host_args=('--ServerApp.local_hostnames=["localhost"]')
if [[ -n "${CODESPACE_NAME:-}" && -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]]; then
  # Permit this Codespace's private forwarded hostname while retaining host checks.
  notebook_hosts="$("$notebook_python" -c 'import json, os, sys; print(json.dumps(["localhost", os.environ["CODESPACE_NAME"] + "-" + sys.argv[1] + "." + os.environ["GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"]]))' "$notebook_port")"
  notebook_host_args=("--ServerApp.local_hostnames=$notebook_hosts")
fi
echo "Starting Jupyter with $notebook_python"
echo "Notebook: notebooks/local_sandbox.ipynb"
echo "In Codespaces: Ports → $notebook_port → Open in Browser (keep the port private)."
echo "If prompted for a token, use the token printed by Jupyter below."
echo "Stop with Ctrl+C twice. Use PORT=8889 bash scripts/notebook.sh for a different port."

# An explicit repository root keeps ../runs/ result links inside the served tree.
# Keep Jupyter's token authentication enabled and bind only to loopback.
exec "$notebook_python" -m jupyterlab \
  --no-browser \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.port="$notebook_port" \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$notebook_root" \
  --LabApp.default_url=/lab/tree/notebooks/local_sandbox.ipynb \
  "${notebook_host_args[@]}" \
  "$@"
