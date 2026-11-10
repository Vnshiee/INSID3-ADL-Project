#!/usr/bin/env bash
# Create the Python environment on the GPU server. Safe to re-run.
#   ./setup_server.sh                 venv in ./.venv (needs python3.10 or uv)
#   ./setup_server.sh --conda         conda env "insid3" (python 3.10)
#   ./setup_server.sh --with-crf      also build the optional GPU CRF (needs nvcc)
#   ./setup_server.sh --clone-dinov3  clone facebookresearch/dinov3 (hub code only, ~MBs) to $INSID3_DINOV3_REPO
# Reference configuration of the official repo: Python 3.10, PyTorch 2.7.1, CUDA 12.6 wheels.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f env.sh ]] && source env.sh || source env.example.sh
USE_CONDA=0; WITH_CRF=0; CLONE=0
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
for a in "$@"; do case "$a" in
  --conda) USE_CONDA=1;; --with-crf) WITH_CRF=1;; --clone-dinov3) CLONE=1;;
  -h|--help) sed -n '2,8p' "$0"; exit 0;; *) echo "unknown option $a"; exit 1;; esac; done

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
  || echo "WARNING: nvidia-smi not found — is this the GPU server?"

if [[ $USE_CONDA == 1 ]]; then
  conda env create -f environment.yml || conda env update -f environment.yml
  PY="$(conda run -n insid3 which python)"
else
  if command -v uv >/dev/null; then uv venv --python 3.10 .venv; PIP="uv pip install --python .venv/bin/python"
  else python3.10 -m venv .venv; .venv/bin/python -m pip install --upgrade pip; PIP=".venv/bin/python -m pip install"; fi
  $PIP torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"
  $PIP -r requirements-research.txt
  PY="$PWD/.venv/bin/python"
fi

if [[ $WITH_CRF == 1 ]]; then
  mkdir -p third_party && [[ -d third_party/CRF ]] || git clone https://github.com/netw0rkf10w/CRF.git third_party/CRF
  (cd third_party/CRF && "$PY" setup.py install)
fi
if [[ $CLONE == 1 ]]; then
  [[ -d "$INSID3_DINOV3_REPO" ]] || git clone --depth 1 https://github.com/facebookresearch/dinov3.git "$INSID3_DINOV3_REPO"
fi
if [[ ! -f env.sh ]]; then
  cp env.example.sh env.sh
  sed -i.bak "s#INSID3_PYTHON:-python#INSID3_PYTHON:-$PY#" env.sh && rm -f env.sh.bak
  echo "created env.sh (INSID3_PYTHON=$PY) — edit the data/output/weights paths"
fi

"$PY" scripts/verify_upstream.py
"$PY" -m pytest tests -q
"$PY" scripts/check_env.py --datasets || true
echo "Setup finished. Interpreter: $PY"
