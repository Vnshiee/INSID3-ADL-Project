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


def _class_miou(cls, inter, union):
    """Official class-wise mIoU (utils/metrics.py AverageMeter) of a set of episodes.

    For every class: total intersection / total union over its episodes; then the mean over classes.
    cls, inter, union: one entry per episode (foreground only).
    """
    ious = []
    for c in np.unique(cls):
        of_class = cls == c
        ious.append(inter[of_class].sum() / max(union[of_class].sum(), 1.0))
    return 100.0 * float(np.mean(ious))


def compare_segmentation(a_dirs, b_dirs, n_boot, seed):
    """Paired, fold-stratified bootstrap of the mIoU difference B - A.

    One resample = for every fold, draw that fold's episodes WITH replacement (same draw for A and B),
    compute both mIoUs, take the mean over folds (= the dataset mIoU), and store B - A.
    """
    rng = np.random.default_rng(seed)
    folds = [_fold_arrays(_load(da), _load(db)) for da, db in zip(a_dirs, b_dirs)]

    # The real (not resampled) numbers.
    miou_a = np.mean([_class_miou(cls, ia, ua) for cls, ia, ua, ib, ub in folds])
    miou_b = np.mean([_class_miou(cls, ib, ub) for cls, ia, ua, ib, ub in folds])

    # Resampled differences. Draw order (fold by fold, n_boot draws each) is fixed by the seed.
    draws = [[rng.integers(0, len(f[0]), size=len(f[0])) for _ in range(n_boot)] for f in folds]
    deltas = np.zeros(n_boot)
    for i in range(n_boot):
        diff_per_fold = []
        for (cls, ia, ua, ib, ub), fold_draws in zip(folds, draws):
            idx = fold_draws[i]                       # the same episodes for A and B: that is "paired"
            diff_per_fold.append(_class_miou(cls[idx], ib[idx], ub[idx]) - _class_miou(cls[idx], ia[idx], ua[idx]))
        deltas[i] = np.mean(diff_per_fold)

    return {"miou_a": float(miou_a), "miou_b": float(miou_b), "delta": float(miou_b - miou_a),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            # two-sided p: how often the resampled difference lies on the other side of 0 (x 2)
            "p_two_sided": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
            "n_folds": len(a_dirs)}


def compare_correspondence(da, db, n_boot, seed, alpha=0.1):
    ra, rb = _load(da), _load(db)
    common = sorted(set(ra) & set(rb))

    def pck(r):
        d = np.array(r["dist"])
        return float((d <= alpha * r["pck_thresh"]).mean())
    diff = np.array([pck(rb[i]) - pck(ra[i]) for i in common])
    # Bootstrap: draw the image pairs with replacement, average the PCK difference, repeat n_boot times.
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(diff), size=len(diff))
        boots[i] = diff[idx].mean() * 100
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
