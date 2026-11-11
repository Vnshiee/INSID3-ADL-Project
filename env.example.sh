# Copy to env.sh and edit. All run_*.sh scripts source ./env.sh if it exists.
# Nothing in the code base hard-codes a machine path; everything is read from here.

export INSID3_DATA_ROOT="${INSID3_DATA_ROOT:-$PWD/data}"            # datasets (layout: docs/6_setup_and_running.md)
export INSID3_OUTPUT_ROOT="${INSID3_OUTPUT_ROOT:-$PWD/outputs}"     # logs, predictions, metrics
export INSID3_PRETRAIN_DIR="${INSID3_PRETRAIN_DIR:-$PWD/pretrain}"  # DINOv3 .pth files (official names)
# Optional: local clone of facebookresearch/dinov3 for offline torch.hub (setup_server.sh --clone-dinov3)
export INSID3_DINOV3_REPO="${INSID3_DINOV3_REPO:-$PWD/third_party/dinov3}"

# Calibration files written by STAGE=calibrate ./run_experiments.sh
export INSID3_CALIBRATION_PATH="${INSID3_CALIBRATION_PATH:-$INSID3_OUTPUT_ROOT/calibration/calib_large_1024.pt}"
export INSID3_CALIBRATION_PATH_BASE768="${INSID3_CALIBRATION_PATH_BASE768:-$INSID3_OUTPUT_ROOT/calibration/calib_base_768.pt}"
export INSID3_CALIB_IMAGES="${INSID3_CALIB_IMAGES:-$INSID3_DATA_ROOT/COCO2014/train2014}"   # NOT an eval set
export INSID3_CALIB_NUM="${INSID3_CALIB_NUM:-500}"

# Interpreter (e.g. .venv/bin/python or a conda env python)
export INSID3_PYTHON="${INSID3_PYTHON:-python}"
# GPU selection
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Which segmentation benchmarks the *test* stage runs (space separated; official list below)
export DATASETS="${DATASETS:-coco lvis pascal_part paco_part isaid isic lung suim permis}"
