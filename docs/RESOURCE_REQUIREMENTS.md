# Resource requirements

Labels: **[official]** stated by the INSID3 repo/paper, **[measured]** measured here, **[estimate]** reasoned
but unmeasured, **[unknown]**.

## Software

| Item | Requirement |
|---|---|
| Python | 3.10 [official]; code avoids 3.11+ features; tests executed with 3.10.20 [measured] |
| PyTorch | 2.7.1 + torchvision 0.22.1 [official]; CUDA 12.6 wheels (`cu126`) [official] |
| NVIDIA driver | must support CUDA 12.6 (driver ≥ 560 on Linux) [estimate — check `nvidia-smi`] |
| Other | numpy ≥ 2, scikit-learn ≥ 1.5, einops, pycocotools, Pillow, tqdm, matplotlib, PyYAML, pytest (requirements-research.txt) |
| Optional | `CRF` (github.com/netw0rkf10w/CRF) for CRF refinement — needs a CUDA toolchain to build |
| Data tools | wget, curl, unzip, tar, git, gdown 5.2.0, pandas (ISIC), Kaggle access (Chest X-ray, SUIM) |
| Network at run time | none if `INSID3_DINOV3_REPO` points to a local clone; otherwise torch.hub contacts GitHub |

## Hardware

| Item | Value |
|---|---|
| GPU used by the authors | RTX 4090 and RTX 3090 (24 GB) [official, Supp. D] |
| GPU memory, ViT-L @ 1024, 1-shot | **[measured]** 1,548–1,564 MB peak VRAM on RTX 3070 8 GB (`results/summary.md`) |
| 5-shot | **[measured]** 2,258–2,260 MB peak VRAM on RTX 3070 8 GB |
| CPU / RAM | agglomerative clustering on CPU (4096² distance matrix, ~130 MB in float64) [estimate]; executed on single workstation |
| Runtime | 302 ms/episode incl. CRF, 247 ms without, RTX 4090 [official]; **[measured]** 544–555 ms/episode (1-shot) and 1,190–1,205 ms/episode (5-shot) on RTX 3070 8 GB |

## Storage

| Item | Size |
|---|---|
| DINOv3 ViT-L/16 weights | ~1.2 GB [estimate: 300 M params × 4 B] |
| DINOv3 ViT-B/16 weights | ~0.35 GB [estimate] |
| COCO 2014 train+val + annotations | ~20 GB [estimate] |
| COCO 2017 (LVIS, PACO-Part) | ~19 GB [estimate] |
| PASCAL VOC 2010 + parts | ~1.5 GB [estimate] |
| ISIC 2018 Task 1 | ~11 GB [estimate] |
| Chest X-ray | ~4.5 GB [estimate] |
| SUIM | < 1 GB [estimate] |
| iSAID-5ⁱ patches | [unknown] |
| PerMIS (requires TAO test frames) | [unknown, large] — download temporarily, generated set is small |
| SPair-71k | ~0.3 GB [estimate] |
| Outputs | ~0.3 KB per episode in JSONL; predictions only if `run.save_predictions > 0`; calibration file ~ 20–40 MB [estimate] |
| Temporary space during preparation | zip + extracted copies: plan ≥ 2× dataset size |

Recommended free space for everything: ≥ 150 GB [estimate].

## Inputs that must be obtained by you

1. DINOv3 weights — licence form at github.com/facebookresearch/dinov3 (cannot be automated).
2. Dataset downloads (some require Google Drive or Kaggle access).
3. A GPU server with a CUDA 12.6-compatible driver.
