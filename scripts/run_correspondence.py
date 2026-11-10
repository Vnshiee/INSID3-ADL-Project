"""Evaluate semantic correspondence on SPair-71k (official PCK protocol).

Examples
    # README reference setting (ViT-B, 768 px, s = 20), original vs debiased features:
    python scripts/run_correspondence.py --config configs/correspondence/base_readme.yaml \
        --set correspondence.debiased=false run.exp_name=corr_original
    python scripts/run_correspondence.py --config configs/correspondence/base_readme.yaml
"""
import argparse

import _bootstrap  # noqa: F401

from insid3_ext.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    from insid3_ext.experiment import run_correspondence
    run_correspondence(cfg)


if __name__ == "__main__":
    main()
