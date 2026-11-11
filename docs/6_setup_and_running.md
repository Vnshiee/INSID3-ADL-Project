# 6. Setup and running

## 6.1 Requirements

* Linux with an NVIDIA GPU and a driver for CUDA 12.6. 8 GB is enough: we measured 1.5 GB peak at 1-shot
  and 2.3 GB at 5-shot (ViT-L, 1024 px, RTX 3070).
* Python 3.10, PyTorch 2.7.1, torchvision 0.22.1 (the official reference versions).
* Disk: about 25 GB for COCO 2014 + PASCAL-Part + SPair-71k; about 150 GB for all 9 benchmarks.
* Time on an RTX 3070: about 0.55 s per 1-shot episode and 1.2 s per 5-shot episode. Most of it is the
  CPU clustering step.

## 6.2 Environment

```bash
./setup_server.sh                  # venv in .venv (or: --conda); --with-crf only for the CRF stage
cp env.example.sh env.sh           # then edit the paths in env.sh
```

`env.sh` holds every machine-specific path (data, outputs, weights, calibration files, Python). Nothing in
the code or configs contains a fixed path.

## 6.3 DINOv3 weights

Request access at github.com/facebookresearch/dinov3 (licence form), then:

```bash
DINOV3_URL_LARGE='<url from the e-mail>' DINOV3_URL_BASE='<url>' ./download_weights.sh large base
```

The files keep their official names (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, …) and go to
`$INSID3_PRETRAIN_DIR`.

## 6.4 Datasets

```bash
./prepare_datasets.sh --yes coco pascal_part spair
python scripts/check_env.py --datasets coco pascal_part spair
```

The folder layout is the official one (`docs/data.md`).

## 6.5 Running

```bash
STAGE=sanity ./run_experiments.sh        # tests, official files unchanged, data, 20-episode smoke run
STAGE=calibrate ./run_experiments.sh     # positional calibration (minutes)
```

Then the stages listed at the top of `run_experiments.sh` (round 1: `r1_tune`, `r1_combined`, `r1_test`,
`r1_corr`, `r1_report`; round 2: `bias`, `t1`, `t2`, `corr`, `attribution`, `report`). Run long stages
inside `tmux`. An interrupted stage continues where it stopped when started again.

A single run by hand:

```bash
python scripts/run_segmentation.py --config configs/baseline.yaml --set data.dataset=coco data.fold=0
python scripts/run_segmentation.py --config configs/task1/calibrated.yaml --folds all --set run.exp_name=my_test
python scripts/run_correspondence.py --config configs/correspondence/large.yaml --set debias.mode=token
```

`--set key=value` changes any setting of `insid3_ext/config.py`.

## 6.6 Figures for the report

```bash
python scripts/visualize.py calibration --calibration $INSID3_CALIBRATION_PATH --out figs/
python scripts/visualize.py similarity --configs configs/baseline.yaml configs/task1/token.yaml \
    --ref-image r.jpg --ref-mask r.png --target t.jpg --out figs/similarity.png
python scripts/visualize.py prototypes --config configs/task2/lse.yaml \
    --ref-image r.jpg --ref-mask r.png --target t.jpg --out figs/prototypes.png
```

* `calibration`: the positional share ρ per direction, the λ of every mode, the per-position energy map and the
  first positional patterns (the figures in `results/round*/calibration/`).
* `similarity`: like paper Fig. 4, the similarity of the reference object to every target patch, with original
  features and for each given config. This shows the same-position bias, and how much of it each mode removes.
* `prototypes`: which reference patches belong to which prototype, the similarity map of each prototype, and the
  combined map.

## 6.7 Common problems

| Message / symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() is False` | CPU-only PyTorch or driver too old | `nvidia-smi`; reinstall torch from the cu126 index |
| `DINOv3 weights not found` | wrong `INSID3_PRETRAIN_DIR` or file name | keep the official file names |
| torch.hub HTTP error | no internet on the server | `./setup_server.sh --clone-dinov3` and set `INSID3_DINOV3_REPO` |
| `Calibration … was computed for model_size=…` | calibration of another backbone or resolution | run `STAGE=calibrate` for this setting |
| `debias.mode=… needs debias.calibration_path` | calibration missing | run `STAGE=calibrate` first |
| `Episode i fingerprint mismatch on resume` | data, seed or code changed during a run | `--set run.resume=false` or a new `run.exp_name` |
| `ModuleNotFoundError: CRF` | optional package not built | use bilinear, or `./setup_server.sh --with-crf` |
| very slow, GPU idle | CPU clustering is the bottleneck | do not oversubscribe the CPU (`OMP_NUM_THREADS`) |
| `verify_upstream.py` FAILED | an official file was edited | restore it from the official repository |

Cross-check with the official script (same episodes, same model; it needs `pretrain/` in the working folder):

```bash
python inference_segmentation.py --dataset coco --fold 0 --data-root $INSID3_DATA_ROOT --exp-name official_check
```

Its mIoU must equal our `baseline` on COCO fold 0.
