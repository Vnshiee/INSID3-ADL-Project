"""Are two runs of the SAME configuration on two machines interchangeable?

Compares a run made on this machine (--new) with the same run made elsewhere (--ref, e.g. copied from
another GPU). Episodes are matched by index and must have identical fingerprints (same episodes).

Decision rule (fixed in advance):
    PASS  if  |mIoU_new − mIoU_ref| ≤ 0.05  and  ≥ 98 % of episodes have identical IoU (|Δ| < 1e-6)
If PASS, runs copied from the reference machine can be paired with new runs in compare_runs.py /
compare_all.py. If FAIL, every comparison must use runs from one machine only (re-run the baselines here).

    python scripts/check_consistency.py --ref <ref run dir> --new <new run dir>
Exit code: 0 = PASS, 2 = FAIL, 1 = not comparable (different episodes / missing files).
"""
import argparse
import json
import os
import sys

import _bootstrap  # noqa: F401

from insid3_ext.experiment import compute_seg_metrics, read_jsonl

MAX_MIOU_DIFF = 0.05
MIN_IDENTICAL = 0.98


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    ra = {r["idx"]: r for r in read_jsonl(os.path.join(args.ref, "episodes.jsonl"))}
    rb = {r["idx"]: r for r in read_jsonl(os.path.join(args.new, "episodes.jsonl"))}
    common = sorted(set(ra) & set(rb))
    if not common:
        print("NOT COMPARABLE: no common episodes (missing episodes.jsonl?)")
        return 1
    if any(ra[i]["fingerprint"] != rb[i]["fingerprint"] for i in common):
        print("NOT COMPARABLE: episode fingerprints differ (different data or seed)")
        return 1

    deltas = [abs(rb[i]["fg_iou"] - ra[i]["fg_iou"]) for i in common]
    identical = sum(d < 1e-6 for d in deltas) / len(deltas)
    class_ids = sorted({ra[i]["class_id"] for i in common})
    ma = compute_seg_metrics([ra[i] for i in common], class_ids, "coco")["miou"]
    mb = compute_seg_metrics([rb[i] for i in common], class_ids, "coco")["miou"]
    passed = abs(ma - mb) <= MAX_MIOU_DIFF and identical >= MIN_IDENTICAL
    rep = {"episodes": len(common), "miou_ref": ma, "miou_new": mb, "miou_diff": mb - ma,
           "identical_fraction": identical, "max_abs_iou_diff": max(deltas),
           "rule": f"|dmIoU| <= {MAX_MIOU_DIFF} and identical >= {MIN_IDENTICAL}",
           "verdict": "PASS" if passed else "FAIL"}
    print(json.dumps(rep, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump(rep, fp, indent=2)
    print("PASS: runs from the two machines are interchangeable." if passed else
          "FAIL: do not pair runs across machines (each stage computes its own baseline).")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
