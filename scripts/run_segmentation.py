"""Evaluate in-context segmentation (official episodes and mIoU).

Examples
    python scripts/run_segmentation.py --config configs/baseline.yaml
    python scripts/run_segmentation.py --config configs/baseline.yaml --folds all
    python scripts/run_segmentation.py --config configs/task1/calibrated.yaml \
        --set data.dataset=pascal_part run.exp_name=t1_calibrated
"""
import argparse
import json

import _bootstrap  # noqa: F401

from insid3_ext.config import DATASET_FOLDS, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--folds", default=None,
                    help="'all' or comma list (e.g. 0,2). Default: data.fold from the config.")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.folds is None:
        folds = [cfg.data.fold]
    elif args.folds == "all":
        folds = list(range(DATASET_FOLDS.get(cfg.data.dataset, 1))) if cfg.data.split == "test" else [0]
    else:
        folds = [int(f) for f in args.folds.split(",")]

    from insid3_ext.builder import build_model
    from insid3_ext.experiment import run_segmentation
    from insid3_ext.runtime import resolve_device, set_seed

    cfg.model.device = resolve_device(cfg.model.device)
    set_seed(cfg.run.seed)
    model = build_model(cfg)            # built once, reused across folds (training-free, stateless)
    results = []
    for f in folds:
        fold_cfg = load_config(args.config, args.set + [f"data.fold={f}"])
        fold_cfg.model.device = cfg.model.device
        results.append(run_segmentation(fold_cfg, model=model))
    if len(results) > 1:
        mean = sum(r["miou"] for r in results) / len(results)
        print(json.dumps({"folds": folds, "miou_per_fold": [round(r["miou"], 2) for r in results],
                          "miou_mean": round(mean, 2)}, indent=2))


if __name__ == "__main__":
    main()
