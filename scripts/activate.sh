# Source this file from the repository root: source scripts/activate.sh
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TORCH_HOME="$PWD/runtime/cache/torch"
export XDG_CACHE_HOME="$PWD/runtime/cache"
export TMPDIR="$PWD/runtime/tmp"
