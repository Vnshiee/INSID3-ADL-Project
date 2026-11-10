"""Task 1 analysis + calibration: measure positional components on natural images.

Produces <out>.pt (used by debias modes calibrated / image / token) and <out>_summary.json.
Plots are made separately with  scripts/visualize.py calibration --calibration <out>.pt

The images must NOT come from an evaluation set. Default recommendation: a random
subset of COCO train2014 (disjoint from the val2014 images of COCO-20i test episodes).

Example
    python scripts/calibrate_positional.py --config configs/base.yaml \
        --image-dir $INSID3_DATA_ROOT/COCO2014/train2014 --num-images 500 \
        --out $INSID3_OUTPUT_ROOT/calibration/calib_large_1024.pt
"""
import argparse
import glob
import json
import os
import random

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from insid3_ext.config import load_config

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(image_dir, image_list, num, seed):
    if image_list:
        with open(image_list) as fp:
            paths = [l.strip() for l in fp if l.strip()]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
                       if p.lower().endswith(EXTS))
    if not paths:
        raise SystemExit("No calibration images found.")
    rng = random.Random(seed)                 # local RNG
    rng.shuffle(paths)
    return paths[:num]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model section is used (size, image_size, s)")
    ap.add_argument("--set", nargs="*", default=[])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-dir")
    g.add_argument("--image-list", help="text file with one image path per line")
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # The calibration itself must use the official (hard) model construction.
    cfg = load_config(args.config, args.set + ["debias.mode=hard", "debias.calibration_path=null",
                                               "prototypes.enabled=false"])
    from insid3_ext.builder import build_model
    from insid3_ext.calibration import PositionalStatsAccumulator, summarize_calibration
    from insid3_ext.runtime import environment_report, resolve_device
    from utils.data import build_transform

    device = resolve_device(cfg.model.device)
    model = build_model(cfg).to(device).eval()
    svd = model._pos_svd
    acc = PositionalStatsAccumulator(svd.U, svd.coeff, cfg.model.svd_components, svd.grid)
    transform = build_transform(cfg.model.image_size)

    paths = list_images(args.image_dir, args.image_list, args.num_images, args.seed)
    for i in tqdm(range(0, len(paths), args.batch_size), ncols=80):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i + args.batch_size]])
        with torch.no_grad():
            f = model._extract_features(batch.unsqueeze(0).to(device))[0]   # (N, C, h, w), as inference
            acc.update(F.normalize(f, p=2, dim=1))

    meta = {"model_size": cfg.model.model_size, "image_size": cfg.model.image_size,
            "seed": args.seed, "image_source": args.image_dir or args.image_list,
            "env": environment_report()}
    cal = acc.finalize(meta=meta, S=svd.S)
    cal["image_paths"] = paths
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cal, args.out)
    summary = summarize_calibration(cal, cfg.model.svd_components)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
