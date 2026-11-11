# INSID3: Adaptive Positional Debiasing and Multi-Prototype Matching

This repository is a research extension of **INSID3: Training-Free In-Context Segmentation with DINOv3**
(Cuttano et al., CVPR 2026, [official code](https://github.com/visinf/INSID3)). The assignment is in
[`topic.md`](topic.md).

INSID3 segments an object in a new image, given one or a few example images with a mask, using a frozen
DINOv3 network and no training. We studied two changes to it:

1. **Task 1 — Adaptive positional debiasing.** INSID3 deletes 500 "positional" directions from every
   feature. We remove each direction only as much as it is really positional (weighted, spectral,
   calibrated, per-image and per-position variants).
2. **Task 2 — Multi-prototype reference matching.** INSID3 describes the reference object by one average
   vector. We describe it by several vectors (one per part) and match against all of them (max, weighted,
   log-sum-exp, Chamfer, optimal transport).

## Main findings

| | Result |
|---|---|
| Positional debiasing at all | essential: −2.3 (COCO test) to −3.6 (COCO tuning) mIoU without it |
| Task 1, segmentation | no adaptive mode beat the original hard projection; the round-1 "win" (+0.44) was selection noise (−0.08 with 4,000 new episodes, ±0.00 on the test sets) |
| Task 1, semantic correspondence | removing only the positional part is clearly better: **+2.4 PCK@0.10** [+2.15, +2.68] on SPair-71k (ViT-L, 1024 px) |
| Why hard projection works for segmentation | the specific empty-image directions matter (random or PCA directions fail, and centring fails); the first 5 directions give ~75 % of the gain |
| Task 2 | more prototypes never helped in 1-shot (−0.31 for the best one, with 4,000 episodes); with 5 shots, per-shot prototypes caused the round-1 loss (−1.81 PASCAL-Part), and pooling the shots removes most of it |

Everything, with confidence intervals and limitations: [`docs/5_results.md`](docs/5_results.md).

**Compute note.** Code development and the first round of experiments ran on an RTX 3070 (8 GB). The
second-round benchmarks were computed on a larger GPU server for speed. Every run records its hardware in
`env.json`. Methods are only compared with a baseline from the same machine, because different GPUs give
slightly different floating-point results.

## Documentation

| | |
|---|---|
| [`docs/1_background.md`](docs/1_background.md) | the task, DINOv3, how INSID3 works, paper vs code |
| [`docs/2_method.md`](docs/2_method.md) | our two methods and the maths behind them |
| [`docs/3_code_walkthrough.md`](docs/3_code_walkthrough.md) | the official code first, then every change we made, and how we test it |
| [`docs/4_experiments.md`](docs/4_experiments.md) | protocol: tuning split, selection, paired bootstrap, what was run |
| [`docs/5_results.md`](docs/5_results.md) | results of both rounds and what they mean |
| [`docs/6_setup_and_running.md`](docs/6_setup_and_running.md) | installation, data, weights, running, common problems |
| [`docs/7_reading_notes.md`](docs/7_reading_notes.md) | notes on the papers we used |

## Repository layout

```
models/ utils/ datasets/ inference_*.py opts.py   official INSID3 code, unchanged (scripts/verify_upstream.py)
upstream/                                         official README and checksums
insid3_ext/                                       our code
    model.py          INSID3Ext: the official model with our two changes plugged in
    debiasing.py      Task 1: the debiasing modes
    calibration.py    Task 1: measuring the positional components on real images
    prototypes.py     Task 2: building several prototypes
    matching.py       Task 2: scoring patches and regions against the prototypes
    config.py         every setting, with the reason for its default value
    experiment.py     evaluation loop with per-episode logs, resume and episode fingerprints
    builder.py, runtime.py, datasets_ext.py   model loading, seeding/timing, COCO tuning split
configs/            experiment settings (base.yaml = official values)
    selected/         settings chosen in round 1;   selected_rev/   chosen again in round 2
scripts/            run, calibrate, measure bias, select, compare, aggregate, plot, checks
tests/              153 tests (CPU, no weights needed)
results/round1/     tables and figures of round 1;   results/round2/   of round 2
run_experiments.sh  every experiment, as stages
```

## Quick start

```bash
./setup_server.sh && cp env.example.sh env.sh      # then edit the paths in env.sh
python -m pytest tests -q                          # 153 passed
python scripts/verify_upstream.py                  # official files unchanged
STAGE=sanity ./run_experiments.sh                  # data check + 20-episode smoke run on the GPU
```

Details: [`docs/6_setup_and_running.md`](docs/6_setup_and_running.md).

## License and attribution

* Original INSID3 implementation © 2026 Claudia Cuttano et al. (Apache-2.0).
* Research extensions, experiments and documentation © 2026 Vnshiee (Apache-2.0). See [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE).

```bibtex
@inproceedings{cuttano2026insid3,
  title     = {{INSID3}: Training-Free In-Context Segmentation with {DINOv3}},
  author    = {Claudia Cuttano and Gabriele Trivigno and Christoph Reich and Daniel Cremers and Carlo Masone and Stefan Roth},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
