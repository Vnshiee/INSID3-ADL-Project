"""Paired bootstrap comparison of two methods evaluated on the SAME episodes.

Segmentation: give one run directory per fold for A and for B (same order). Episodes
are resampled with replacement *within each fold*, class-wise mIoU is recomputed for
both methods on the same resample, and the statistic is the mean-over-folds mIoU
difference (B − A). Fingerprints are checked so that only identical episodes are paired.

Correspondence: per-pair image-wise PCK differences are bootstrapped.

    python scripts/compare_runs.py --a out/baseline/coco/fold0_shot1_seed0 \
                                   --b out/t1_calibrated/coco/fold0_shot1_seed0
"""
import argparse
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from insid3_ext.experiment import read_jsonl


def _load(d):
    recs = read_jsonl(os.path.join(d, "episodes.jsonl"))
    if not recs:
        raise SystemExit(f"No episodes in {d}")
    return {r["idx"]: r for r in recs}


def _fold_arrays(ra, rb):
    common = sorted(set(ra) & set(rb))
    bad = [i for i in common if ra[i].get("fingerprint") != rb[i].get("fingerprint")]
    if bad:
        raise SystemExit(f"{len(bad)} episodes differ between runs (e.g. idx {bad[0]}): not paired.")
    cls = np.array([ra[i]["class_id"] for i in common])
    ia = np.array([ra[i]["inter"][1] for i in common], float)
    ua = np.array([ra[i]["union"][1] for i in common], float)
    ib = np.array([rb[i]["inter"][1] for i in common], float)
    ub = np.array([rb[i]["union"][1] for i in common], float)
    return cls, ia, ua, ib, ub


def _miou(counts, onehot, inter, union):
    """counts (B, E) resample multiplicities → (B,) class-wise mIoU (official formula)."""
    I = counts @ (onehot * inter[:, None])
    U = counts @ (onehot * union[:, None])
    present = (counts @ onehot) > 0
    iou = I / np.maximum(U, 1.0)
    return 100.0 * (iou * present).sum(1) / np.maximum(present.sum(1), 1)


def compare_segmentation(a_dirs, b_dirs, n_boot, seed):
    rng = np.random.default_rng(seed)
    point_a, point_b, boots = [], [], []
    for da, db in zip(a_dirs, b_dirs):
        cls, ia, ua, ib, ub = _fold_arrays(_load(da), _load(db))
        classes, inv = np.unique(cls, return_inverse=True)
        onehot = np.eye(len(classes))[inv]
        E = len(cls)
        ones = np.ones((1, E))
        point_a.append(_miou(ones, onehot, ia, ua)[0])
        point_b.append(_miou(ones, onehot, ib, ub)[0])
        idx = rng.integers(0, E, size=(n_boot, E))
        counts = np.zeros((n_boot, E))
        np.add.at(counts, (np.arange(n_boot)[:, None], idx), 1)
        boots.append(_miou(counts, onehot, ib, ub) - _miou(counts, onehot, ia, ua))
    delta = np.mean(boots, axis=0)
    return {"miou_a": float(np.mean(point_a)), "miou_b": float(np.mean(point_b)),
            "delta": float(np.mean(point_b) - np.mean(point_a)),
            "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))),
            "n_folds": len(a_dirs)}


def compare_correspondence(da, db, n_boot, seed, alpha=0.1):
    ra, rb = _load(da), _load(db)
    common = sorted(set(ra) & set(rb))

    def pck(r):
        d = np.array(r["dist"])
        return float((d <= alpha * r["pck_thresh"]).mean())
    diff = np.array([pck(rb[i]) - pck(ra[i]) for i in common])
    rng = np.random.default_rng(seed)
    chunks = []                                   # chunked: avoids an (n_boot x n_pairs) index array
    for start in range(0, n_boot, 250):
        b = min(250, n_boot - start)
        chunks.append(diff[rng.integers(0, len(diff), size=(b, len(diff)))].mean(1) * 100)
    boots = np.concatenate(chunks)
    pa = float(np.mean([pck(ra[i]) for i in common]) * 100)
    pb = float(np.mean([pck(rb[i]) for i in common]) * 100)
    return {"alpha": alpha, "pck_a": pa, "pck_b": pb, "delta_pck": float(diff.mean() * 100),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))),
            "n_pairs": len(common)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, help="run dir(s) of method A (one per fold)")
    ap.add_argument("--b", nargs="+", required=True, help="run dir(s) of method B (same fold order)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pck-alpha", type=float, default=0.1)
    args = ap.parse_args()
    if len(args.a) != len(args.b):
        raise SystemExit("--a and --b need the same number of run directories")
    first = read_jsonl(os.path.join(args.a[0], "episodes.jsonl"))
    if first and "dist" in first[0]:
        res = compare_correspondence(args.a[0], args.b[0], args.bootstrap, args.seed, args.pck_alpha)
    else:
        res = compare_segmentation(args.a, args.b, args.bootstrap, args.seed)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
