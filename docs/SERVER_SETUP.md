# Server setup and execution

## 0. Transfer

Clone the repository onto a CUDA-capable Linux machine (or transfer the workspace). All paths inside configs resolve from `env.sh`.

## 1. Environment

```bash
cd ~/INSID3_Research
./setup_server.sh --clone-dinov3            # venv (.venv) + torch 2.7.1/cu126 + deps + DINOv3 hub code
# or: ./setup_server.sh --conda --clone-dinov3
# optional CRF: ./setup_server.sh --with-crf
```

The script also runs `scripts/verify_upstream.py`, the CPU test suite and `scripts/check_env.py`.
It creates `env.sh` from `env.example.sh` on first run. Edit it:

```bash
export INSID3_DATA_ROOT=/data/insid3            # datasets
export INSID3_OUTPUT_ROOT=/scratch/insid3_out   # results
export INSID3_PRETRAIN_DIR=/data/insid3/pretrain
export INSID3_DINOV3_REPO=$HOME/INSID3_Research/third_party/dinov3
export INSID3_PYTHON=$HOME/INSID3_Research/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
```

## 2. Weights (manual licence step)

Request access at https://github.com/facebookresearch/dinov3, then:

```bash
DINOV3_URL_LARGE='<url from e-mail>' DINOV3_URL_BASE='<url>' ./download_weights.sh large base
```

Expected files (official names): `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`,
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` in `$INSID3_PRETRAIN_DIR`.

## 3. Datasets

```bash
./prepare_datasets.sh --dry-run coco spair     # inspect commands first
./prepare_datasets.sh coco spair pascal_part   # minimum for Tasks 1/2 development
./prepare_datasets.sh lvis paco_part isic lung suim isaid permis   # full benchmark (large)
python scripts/check_env.py --datasets coco spair pascal_part
```

Expected layout under `$INSID3_DATA_ROOT` (identical to official docs/data.md):

```
COCO2014/{train2014,val2014,annotations/{train2014,val2014},splits/{trn,val}}
LVIS/{coco/{train2017,val2017},lvis_train.pkl,lvis_val.pkl}
PACO-Part/{coco -> ../LVIS/coco, paco/{paco_part_train.pkl,paco_part_val.pkl}}
Pascal-Part/VOCdevkit/VOC2010/{JPEGImages,Annotations_Part_json_merged_part_classes,all_obj_part_to_image.json}
ISIC/{ISIC2018_Task1_Training_GroundTruth,ISIC2018_Task1-2_Training_Input/{1,2,3}}
LungSegmentation/{CXR_png,masks}
SUIM/{images,masks}
iSAID/{val/{images,semantic_png},splits/val}
PerMIRS/<video id>/{0,1,2}.jpg + masks.npz.npy
SPair-71k/{JPEGImages,PairAnnotation,ImageAnnotation,Layout/large}
```

## 4. Smoke test on the GPU (minutes)

```bash
source env.sh
$INSID3_PYTHON scripts/run_segmentation.py --config configs/baseline.yaml \
    --set run.max_episodes=20 run.exp_name=smoke run.save_predictions=5
$INSID3_PYTHON scripts/run_segmentation.py --config configs/task2/max.yaml \
    --set run.max_episodes=20 run.exp_name=smoke_t2
```

Check `$INSID3_OUTPUT_ROOT/smoke/coco/fold0_shot1_seed0/{log.txt,metrics.json,predictions/}`.

## 5. Execution order

```bash
STAGE=calibrate ./run_task1.sh     # positional statistics (minutes)
STAGE=tune ./run_baseline.sh       # reference on tuning episodes
STAGE=tune ./run_task1.sh          # Task-1 sweep  -> configs/selected/task1.yaml
STAGE=tune ./run_task2.sh          # Task-2 sweep  -> configs/selected/task2.yaml, combined.yaml
STAGE=tune ./run_combined.sh       # integration choices
STAGE=test ./run_baseline.sh       # official benchmarks + SPair (original vs hard)
STAGE=corr ./run_task1.sh          # SPair for every Task-1 mode
STAGE=test ./run_task1.sh
STAGE=test ./run_task2.sh
STAGE=test ./run_combined.sh
./evaluate.sh                      # results/summary.md, results/comparisons.md
```

or `./run_all_experiments.sh` for everything. Use `DATASETS="coco pascal_part"` to restrict the test stage.
Long jobs: run inside `tmux`/`screen` or submit through your scheduler; interrupted runs resume.

## 6. Version facts

Python 3.10 · PyTorch 2.7.1 · torchvision 0.22.1 · CUDA 12.6 wheels (official reference configuration).
GPU memory and storage: RESOURCE_REQUIREMENTS.md.
