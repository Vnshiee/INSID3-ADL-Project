# Troubleshooting

| Message / symptom | Cause | Fix |
|---|---|---|
| `model.device is 'cuda' but torch.cuda.is_available() is False` | CPU wheel installed, driver too old, no GPU visible | `nvidia-smi`; reinstall `torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126`; check `CUDA_VISIBLE_DEVICES` |
| `DINOv3 weights not found at …` | wrong `INSID3_PRETRAIN_DIR` or file name | file names must be the official ones (SERVER_SETUP.md §2) |
| torch.hub `HTTP Error` / rate limit | server has no GitHub access | `./setup_server.sh --clone-dinov3` elsewhere, copy `third_party/dinov3`, set `INSID3_DINOV3_REPO` |
| `Calibration … was computed for model_size=…` | calibration file from another backbone/resolution | re-run `STAGE=calibrate ./run_task1.sh` or set `debias.calibration_path` |
| `debias.mode=calibrated needs debias.calibration_path` | calibration missing | run calibration first |
| `Token template has N positions but the feature grid is …` | `token` mode at a resolution different from calibration | calibrate at that `image_size` |
| `knee_rank=… exceeds the number of singular values` | spectral knee > channel count (ViT-S has 384 channels) | lower `debias.knee_rank` |
| `Episode i fingerprint mismatch on resume` | dataset, seed or code changed since the run started | `--set run.resume=false` (old log kept as `.bak`) or a new `run.exp_name` |
| `data.num_workers must be 0` | intentional | workers would change the episode sequence |
| `No usable classes … splits/trn` | COCO split files incomplete | re-download `splits.zip` (prepare_datasets.sh coco) |
| tuning log shows overlap > 0 | trn/val split assumption violated | report it; exclude overlapping images or use another tuning set |
| `ModuleNotFoundError: CRF` | optional package not built | use bilinear, or `./setup_server.sh --with-crf` (needs nvcc) |
| out of GPU memory | small GPU / 5-shot at 1024 px | `--set model.image_size=768` (and recalibrate), one run per GPU |
| very slow episodes, GPU idle | CPU-bound agglomerative clustering | faster CPU; avoid oversubscription (`OMP_NUM_THREADS`), fewer parallel runs |
| baseline far from published numbers | weights / resolution / CRF / data mismatch | EXPERIMENTAL_PROTOCOL.md §6; `verify_upstream.py`; compare with `inference_segmentation.py` (official script) on the same fold |
| `verify_upstream.py` FAILED | an official file was edited | restore it from the archive / upstream commit |

Cross-check against the official script (same episodes, same model):

```bash
python inference_segmentation.py --dataset coco --fold 0 --data-root $INSID3_DATA_ROOT --exp-name official_check
```

(it needs `pretrain/` in the working directory: `ln -s $INSID3_PRETRAIN_DIR pretrain`). Its mIoU must equal
`baseline` fold 0 from `run_segmentation.py`.
