"""Episode-level failure analysis between two runs on identical episodes.

Lists the episodes where method B loses (or gains) most IoU w.r.t. method A and relates
the change to logged diagnostics (GT area, predicted area, #prototypes, #candidate tokens,
seed score). Writes <out>.csv and prints a short summary by GT-area tercile and class.

    python scripts/analyze_failures.py --a <baseline run dir> --b <method run dir> --out results/fail_coco_f0
"""
import argparse
import csv
import os
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from insid3_ext.experiment import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    ra = {r["idx"]: r for r in read_jsonl(os.path.join(args.a, "episodes.jsonl"))}
    rb = {r["idx"]: r for r in read_jsonl(os.path.join(args.b, "episodes.jsonl"))}
    common = [i for i in sorted(set(ra) & set(rb)) if ra[i]["fingerprint"] == rb[i]["fingerprint"]]
    if not common:
        raise SystemExit("no paired episodes")
    rows = []
    for i in common:
        a, b = ra[i], rb[i]
        rows.append({"idx": i, "class_id": a["class_id"], "gt_area": a["gt_area"],
                     "iou_a": a["fg_iou"], "iou_b": b["fg_iou"], "delta": b["fg_iou"] - a["fg_iou"],
                     "pred_area_a": a["pred_area"], "pred_area_b": b["pred_area"],
                     "num_prototypes_b": b.get("num_prototypes"), "candidates_a": a.get("candidate_tokens"),
                     "candidates_b": b.get("candidate_tokens"), "seed_score_b": b.get("seed_score")})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["delta"]))

    d = np.array([r["delta"] for r in rows])
    print(f"paired episodes: {len(rows)}  mean ΔIoU(fg): {100 * d.mean():+.2f}  "
          f"wins/ties/losses: {(d > 1e-6).sum()}/{(abs(d) <= 1e-6).sum()}/{(d < -1e-6).sum()}")
    areas = np.array([r["gt_area"] for r in rows])
    cuts = np.quantile(areas, [1 / 3, 2 / 3])
    for name, m in (("small", areas <= cuts[0]), ("medium", (areas > cuts[0]) & (areas <= cuts[1])),
                    ("large", areas > cuts[1])):
        mean = 100 * d[m].mean() if m.any() else float("nan")
        print(f"  GT area {name:<6}: n={m.sum():4d}  mean ΔIoU {mean:+.2f}")
    over = np.array([r["pred_area_b"] > r["pred_area_a"] for r in rows])
    print(f"  B predicts a larger mask than A in {100 * over.mean():.1f}% of episodes "
          f"(mean ΔIoU there {100 * d[over].mean() if over.any() else float('nan'):+.2f})")
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class_id"]].append(r["delta"])
    worst = sorted(by_cls.items(), key=lambda kv: np.mean(kv[1]))[:5]
    print("  worst classes (class_id: mean ΔIoU):", ", ".join(f"{c}: {100 * np.mean(v):+.1f}" for c, v in worst))
    print(f"  {args.top} worst episodes written first in {args.out}.csv")


if __name__ == "__main__":
    main()
