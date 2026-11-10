"""Pre-flight check on the GPU server: imports, CUDA, weights, hub code, datasets.

    python scripts/check_env.py                 # everything
    python scripts/check_env.py --datasets coco spair
Exit code 1 if a required item is missing. Nothing is downloaded.
"""
import argparse
import importlib
import os
import shutil
import sys

import _bootstrap  # noqa: F401

DATASET_DIRS = {
    "coco": ["COCO2014/train2014", "COCO2014/val2014", "COCO2014/annotations", "COCO2014/splits/val"],
    "lvis": ["LVIS/coco/train2017", "LVIS/lvis_val.pkl"],
    "paco_part": ["PACO-Part/coco", "PACO-Part/paco/paco_part_val.pkl"],
    "pascal_part": ["Pascal-Part/VOCdevkit/VOC2010/JPEGImages"],
    "isic": ["ISIC/ISIC2018_Task1_Training_GroundTruth", "ISIC/ISIC2018_Task1-2_Training_Input"],
    "lung": ["LungSegmentation/CXR_png", "LungSegmentation/masks"],
    "suim": ["SUIM/images", "SUIM/masks"],
    "isaid": ["iSAID/val/images", "iSAID/splits/val"],
    "permis": ["PerMIRS"],
    "spair": ["SPair-71k/JPEGImages", "SPair-71k/PairAnnotation/test", "SPair-71k/Layout/large/test.txt"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--model-size", default="large")
    args = ap.parse_args()
    ok = True

    def report(flag, msg, required=True):
        nonlocal ok
        print(("  [ok]   " if flag else ("  [FAIL] " if required else "  [warn] ")) + msg)
        if required and not flag:
            ok = False

    print(f"Python {sys.version.split()[0]} (official: 3.10)")
    report(sys.version_info[:2] >= (3, 10), "python >= 3.10")
    for mod in ["torch", "torchvision", "numpy", "PIL", "einops", "sklearn", "pycocotools",
                "tqdm", "matplotlib", "yaml"]:
        try:
            m = importlib.import_module(mod)
            report(True, f"import {mod} {getattr(m, '__version__', '')}")
        except ImportError as e:
            report(False, f"import {mod}: {e}")
    try:
        importlib.import_module("CRF")
        report(True, "CRF (optional, needed for model.mask_refiner=crf)", required=False)
    except ImportError:
        report(False, "CRF not installed (only needed for model.mask_refiner=crf)", required=False)

    import torch
    report(torch.cuda.is_available(), f"CUDA available (torch {torch.__version__}, cuda {torch.version.cuda})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"         GPU{i}: {p.name}, {p.total_memory / 1e9:.1f} GB, sm_{p.major}{p.minor}")
            report(p.total_memory / 1e9 >= 10, f"GPU{i} memory >= 10 GB (estimate for ViT-L @1024)",
                   required=False)

    from insid3_ext.builder import WEIGHT_FILES
    wdir = os.environ.get("INSID3_PRETRAIN_DIR", "pretrain")
    w = os.path.join(wdir, WEIGHT_FILES[args.model_size])
    report(os.path.isfile(w), f"DINOv3 {args.model_size} weights: {w}")
    repo = os.environ.get("INSID3_DINOV3_REPO")
    if repo:
        report(os.path.isfile(os.path.join(repo, "hubconf.py")), f"local dinov3 hub repo: {repo}")
    else:
        report(False, "INSID3_DINOV3_REPO unset: torch.hub will fetch facebookresearch/dinov3 from GitHub",
               required=False)

    root = os.environ.get("INSID3_DATA_ROOT", "data")
    for ds in args.datasets if args.datasets is not None else DATASET_DIRS:
        for rel in DATASET_DIRS[ds]:
            report(os.path.exists(os.path.join(root, rel)), f"{ds}: {os.path.join(root, rel)}",
                   required=args.datasets is not None)
    out = os.environ.get("INSID3_OUTPUT_ROOT", "outputs")
    os.makedirs(out, exist_ok=True)
    free = shutil.disk_usage(out).free / 1e9
    report(free > 5, f"free disk at {out}: {free:.1f} GB", required=False)
    print("READY" if ok else "NOT READY (see [FAIL] lines)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
