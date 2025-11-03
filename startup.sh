#!/usr/bin/env bash
set -e

# minimal startup: make env if needed, then run

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. install Miniconda/Anaconda and retry."
  exit 1
fi

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="ml-benchmark"
CONDA_BASE="$(conda info --base)"
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda env create -f "$BASE_DIR/environment.yml"
fi

conda run -n "$ENV_NAME" python "$BASE_DIR/run_benchmark.py"


