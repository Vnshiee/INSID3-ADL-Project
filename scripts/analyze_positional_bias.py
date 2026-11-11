"""Direct measurement of positional bias left by each debiasing mode (round 2).

Segmentation mIoU mixes many effects. This script isolates what Task 1 is supposed to change:
for pairs of unrelated natural images (A, B) it measures, after each debiasing mode,

  pos_gap      mean_p cos(x_A(p), x_B(p)) − mean_p cos(x_A(p), x_B(q(p)))   (q(p) ≥ 8 tokens away)
               > 0 means tokens at the SAME position look more similar than at different positions,
               i.e. residual positional bias (0 = none).
  nn_same_pos  fraction of tokens of A whose nearest neighbour in B lies within 1 token of the same
               position (chance ≈ 9 / P).  This is the failure mode shown in paper Fig. 4.
  intra_corr   Pearson correlation between the intra-image cosine-similarity matrix of A before and
               after debiasing (512 random tokens): how much of the image's own structure survives.
  energy_kept  mean squared norm of the debiased token BEFORE re-normalisation (1 = nothing removed).

Images must be disjoint from the calibration set: pass the calibration file and its image list is
excluded automatically.

    python scripts/analyze_positional_bias.py --config configs/base.yaml \
        --image-dir $INSID3_DATA_ROOT/COCO2014/train2014 --calibration $INSID3_CALIBRATION_PATH \
        --pairs 100 --out results/round2/positional_bias \
        --variant hard: --variant none:debias.mode=none \
        --variant hard_s20:model.svd_components=20 \
        --variant spectral_k500_g4:debias.mode=spectral,debias.knee_rank=500,debias.sharpness=4 \
        --variant calibrated:debias.mode=calibrated --variant rho0.3:debias.mode=rho_threshold
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

from insid3_ext.config import load_config

EXTS = (".jpg", ".jpeg", ".png")


def parse_variant(text):
    name, _, ov = text.partition(":")
    return name, [o for o in ov.split(",") if o]


@torch.no_grad()
def pre_norm_debias(deb, f):
    """Debiased features before re-normalisation, (1,T,C,h,w) → (1,T,C,h,w)."""
    return f if deb is None else deb(f, normalize=False)


class _Centered:
    """Apply x ← normalize(x − μ) before a debiaser (mirrors INSID3Ext with debias.center)."""

    def __init__(self, deb, mu):
        self.deb, self.mu = deb, mu.float()

    def __call__(self, f, normalize=True):
        f = F.normalize(f - self.mu.to(f).view(1, 1, -1, 1, 1), p=2, dim=2)
        return self.deb(f, normalize=normalize)


@torch.no_grad()
def metrics_for_pair(fa, fb, h, w, gen):
    """fa, fb: (C, P) unit-norm tokens of two images."""
    C, P = fa.shape
    same = (fa * fb).sum(0).mean()
    ys, xs = torch.div(torch.arange(P), w, rounding_mode="floor"), torch.arange(P) % w
    shift = torch.randperm(P, generator=gen)
    far = ((ys - ys[shift]).abs() + (xs - xs[shift]).abs()) >= 8
    diff = (fa[:, far] * fb[:, shift[far]]).sum(0).mean()
    nn = (fa.T @ fb).argmax(1).cpu()
    near = ((ys - ys[nn]).abs() <= 1) & ((xs - xs[nn]).abs() <= 1)
    return float(same - diff), float(near.float().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--calibration", default=None, help="calibration .pt (its images are excluded)")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", action="append", required=True,
                    help="name:key=val,key=val (debias/model overrides); 'none' mode = no debiasing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from insid3_ext.builder import build_encoder, build_model
    from insid3_ext.config import needs_calibration
    from insid3_ext.debiasing import PositionalDebiaser, load_calibration
    from insid3_ext.debiasing import estimate_positional_svd
    from insid3_ext.runtime import resolve_device, set_seed
    from utils.data import build_transform

    base = load_config(args.config, ["debias.mode=hard", "prototypes.enabled=false"])
    device = resolve_device(base.model.device)
    set_seed(base.run.seed)        # same seed as the experiments -> same (official) positional basis
    enc = build_encoder(base.model.model_size, base.model.weights_path, base.model.dinov3_repo)
    model = build_model(base, encoder=enc).to(device).eval()
    svd = model._pos_svd
    # The basis computed in eval mode (option model.basis_eval_mode); the encoder is in eval mode now.
    svd_eval = estimate_positional_svd(model.encoder, base.model.image_size, device)
    cal = torch.load(args.calibration, map_location="cpu", weights_only=False) if args.calibration else None
    exclude = set(cal.get("image_paths", [])) if cal else set()

    paths = sorted(p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
                   if p.lower().endswith(EXTS) and p not in exclude)
    random.Random(args.seed).shuffle(paths)
    paths = paths[: 2 * args.pairs]
    tf = build_transform(base.model.image_size)

    debs = {}
    for v in args.variant:
        name, ov = parse_variant(v)
        cfg = load_config(args.config, ["prototypes.enabled=false"] + ov)
        if cfg.debias.mode == "none" and not cfg.debias.center:
            debs[name] = None
            continue
        s = cfg.model.svd_components
        c = load_calibration(cfg.debias.calibration_path, cfg.model.model_size, cfg.model.image_size) \
            if needs_calibration(cfg.debias) else None
        dcfg = cfg.debias
        deb = PositionalDebiaser.from_config(dcfg, svd_eval if cfg.model.basis_eval_mode else svd, s, c)
        debs[name] = _Centered(deb, c["feat_mean"]) if dcfg.center else deb

    gen = torch.Generator().manual_seed(0)
    acc = {k: {"pos_gap": [], "nn_same_pos": [], "intra_corr": [], "energy_kept": []} for k in debs}
    for i in range(0, len(paths) - 1, 2):
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 2]]).to(device)
        f = F.normalize(model._extract_features(imgs.unsqueeze(0)), p=2, dim=2).float()  # (1,2,C,h,w)
        _, _, C, h, w = f.shape
        sub = torch.randperm(h * w, generator=gen)[:512]
        orig_intra = (f[0, 0].reshape(C, -1)[:, sub].T @ f[0, 0].reshape(C, -1)[:, sub]).flatten()
        for name, deb in debs.items():
            raw = pre_norm_debias(deb, f)
            d = F.normalize(raw, p=2, dim=2)
            fa, fb = d[0, 0].reshape(C, -1), d[0, 1].reshape(C, -1)
            g, nn = metrics_for_pair(fa, fb, h, w, gen)
            intra = (fa[:, sub].T @ fa[:, sub]).flatten()
            r = torch.corrcoef(torch.stack([orig_intra, intra]))[0, 1]
            acc[name]["pos_gap"].append(g)
            acc[name]["nn_same_pos"].append(nn)
            acc[name]["intra_corr"].append(float(r))
            acc[name]["energy_kept"].append(float(raw.pow(2).sum(2).mean()))

    summary = {}
    for name, m in acc.items():
        summary[name] = {k: (sum(v) / len(v), (torch.tensor(v).std() / len(v) ** 0.5).item())
                         for k, v in m.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as fp:
        json.dump({"pairs": len(paths) // 2, "chance_nn_same_pos": 9 / (h * w), "variants": summary}, fp, indent=2)
    lines = [f"# Residual positional bias per debiasing variant ({len(paths) // 2} held-out image pairs)", "",
             f"Chance level of nn_same_pos: {9 / (h * w):.4f}. Values: mean ± s.e.m.", "",
             "| variant | pos_gap ↓ | nn_same_pos ↓ | intra_corr ↑ | energy_kept |", "|---|---|---|---|---|"]
    for name, m in summary.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[k][0]:.4f} ± {m[k][1]:.4f}"
                                                  for k in ("pos_gap", "nn_same_pos", "intra_corr", "energy_kept")) + " |")
    with open(args.out + ".md", "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
