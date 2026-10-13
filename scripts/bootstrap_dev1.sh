#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runtime/deployment runtime/tmp runtime/cache runtime/runs data
exec 9>runtime/deployment/bootstrap.lock
flock -n 9 || { echo 'An environment installation is already active'; exit 75; }
export TMPDIR="$PWD/runtime/tmp"
export PIP_CACHE_DIR="$PWD/runtime/cache/pip"
export TORCH_HOME="$PWD/runtime/cache/torch"
export PYTHONNOUSERSITE=1
export PIP_PROGRESS_BAR=off
trap 'code=$?; printf "{\"exit_code\": %s}\n" "$code" > runtime/deployment/bootstrap.exit.json' EXIT
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
grep -q 'include-system-site-packages = false' .venv/pyvenv.cfg
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if [[ -f requirements/dev1.lock ]]; then
  .venv/bin/python -m pip install -r requirements/dev1.lock --extra-index-url https://download.pytorch.org/whl/cu121
else
  .venv/bin/python -m pip install -r requirements/dev1.txt
fi
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze --all > runtime/deployment/requirements-resolved.txt
.venv/bin/python -m pepseek doctor > runtime/deployment/doctor.json
.venv/bin/python -c 'import sys; assert sys.prefix != sys.base_prefix; print(sys.executable)'
