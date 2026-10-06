#!/usr/bin/env bash
# Build the Sandbox 1 environment: Python 3.11 venv, HLE from the pinned submodule, this package.
# HLE path taken on 2026-10-05: source build, second attempt (CMake 4 removed <3.5 compatibility; HLE's CMakeLists
# asks for 2.8.11, so CMAKE_POLICY_VERSION_MINIMUM=3.5 is required). The open_spiel wheel fallback was not needed.
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init third_party/hanabi-learning-environment
uv python install 3.11
[ -d .venv ] || uv venv --python 3.11 .venv
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install --python .venv/bin/python ./third_party/hanabi-learning-environment
# the in-tree build leaves artifacts inside the submodule; remove them so the submodule stays clean
rm -rf third_party/hanabi-learning-environment/_skbuild third_party/hanabi-learning-environment/*.egg-info
uv pip install --python .venv/bin/python -e ".[dev,baselines]"
# macOS can flag site-packages/*.pth as hidden, and Python then skips them; sitecustomize.py is a plain import instead.
SP=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
printf 'import sys\n_p = "%s"\nif _p not in sys.path:\n    sys.path.append(_p)\n' "$(pwd)/src" > "$SP/sitecustomize.py"
.venv/bin/python -m ipykernel install --sys-prefix --name culture --display-name "culture (.venv)"
.venv/bin/python -c "import culture, hanabi_learning_environment.pyhanabi; print('environment ok')"
