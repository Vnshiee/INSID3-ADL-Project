"""Figures for the report (all computed from real model outputs / calibration files).

Sub-commands
  calibration   ρ_k, λ_k schedules, energy spectra and the per-token positional energy map
                python scripts/visualize.py calibration --calibration calib.pt --out figs/
  similarity    paper Fig. 4-style reference→target similarity maps for several configs
                python scripts/visualize.py similarity --configs configs/baseline.yaml \
                    configs/task1/calibrated.yaml --ref-image r.jpg --ref-mask r.png \
                    --target t.jpg --out figs/sim.png       (or --ref-point X,Y instead of --ref-mask)
  prototypes    reference prototype assignment + per-prototype target similarity maps
                python scripts/visualize.py prototypes --config configs/task2/max.yaml \
                    --ref-image r.jpg --ref-mask r.png --target t.jpg --out figs/protos.png
"""
import argparse
import os

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from insid3_ext.config import load_config  # noqa: E402


# ───────────────────────────── calibration plots ─────────────────────────────

def plot_calibration(args) -> None:
    from insid3_ext.debiasing import calibrated_gains, spectral_gains, weighted_gains
    cal = torch.load(args.calibration, map_location="cpu", weights_only=False)
    os.makedirs(args.out, exist_ok=True)
    rho, S = cal["rho"].numpy(), cal["S"]
    s = cal["meta"].get("svd_components", 500)
    R = len(rho)
    k = np.arange(1, R + 1)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(k, rho, lw=1)
    ax[0].axvline(s, ls="--", c="gray", label=f"s = {s} (official)")
    ax[0].set(xlabel="component k (constant-image SVD order)", ylabel="ρ_k (positional share)",
              title="Positional share of each component")
    ax[0].legend()
    ax[1].semilogy(k, cal["second_moment"].numpy(), label="feature energy m_k")
    ax[1].semilogy(k, cal["pos_second_moment"].numpy().clip(1e-12), label="positional energy v_k")
    ax[1].axvline(s, ls="--", c="gray")
    ax[1].set(xlabel="k", title="Energy per component (natural images)")
    ax[1].legend()
    ax[2].plot(k, np.ones(R) * (k <= s), label="hard (official)")
    ax[2].plot(k, weighted_gains(R, 0.5).numpy() * (k <= s), label="weighted α=0.5")
    if S is not None:
        for g in (1.0, 4.0):
            ax[2].plot(k, spectral_gains(S, R, s, g).numpy(), label=f"spectral γ={g:g}")
    ax[2].plot(k, calibrated_gains(cal["rho"], R, 1.0).numpy(), label="calibrated β=1")
    ax[2].set(xlabel="k", ylabel="λ_k (fraction removed)", title="Attenuation schedules")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "calibration_spectra.png"), dpi=150)

    h, w = cal["meta"]["grid"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    im = ax[0].imshow(cal["token_energy_mean"].view(h, w).numpy())
    ax[0].set_title("mean ‖U_sᵀx_p‖² per token")
    plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(cal["token_energy_std"].view(h, w).numpy())
    ax[1].set_title("std over images")
    plt.colorbar(im, ax=ax[1])
    corr = cal["per_image_noise_corr"].float().numpy()
    top = min(16, corr.shape[1])
    ax[2].boxplot([corr[:, j] for j in range(top)], showfliers=False)
    ax[2].set(xlabel="component k", ylabel="corr with constant-image template",
              title="Per-image expression of positional components")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "calibration_tokens.png"), dpi=150)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
    for j, comp in enumerate((0, 1, 2, min(9, R - 1))):
        ax[j].imshow(cal["template"][comp].view(h, w).numpy(), cmap="coolwarm")
        ax[j].set_title(f"template π_{comp + 1}(p), ρ={rho[comp]:.2f}")
        ax[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "calibration_templates.png"), dpi=150)
    print(f"saved figures to {args.out}")


# ───────────────────────────── model-based figures ─────────────────────────────

def _features(model, ref_path, tgt_path):
    from utils.data import load_image
    ref, ref_size = load_image(ref_path, model._transform, model.device)
    tgt, _ = load_image(tgt_path, model._transform, model.device)
    with torch.no_grad():
        f = F.normalize(model._extract_features(torch.cat([ref, tgt]).unsqueeze(0)), p=2, dim=2)
        d = model._debias_features(f)
    return f, d, ref_size


def _ref_vector(feats_ref, args, model, ref_size):
    """(C,) reference prototype (mask) or keypoint descriptor."""
    from utils.data import downsample_mask, load_mask
    C, h, w = feats_ref.shape
    if args.ref_point:
        x, y = [float(v) for v in args.ref_point.split(",")]
        px = min(w - 1, int(round(x * w / ref_size[1])))
        py = min(h - 1, int(round(y * h / ref_size[0])))
        return feats_ref[:, py, px]
    m = load_mask(args.ref_mask, model.image_size, model.device).unsqueeze(0)  # (1,1,H,W)
    mk = downsample_mask(m, h, w)
    return F.normalize(feats_ref[:, mk].mean(1), dim=0)


def plot_similarity(args) -> None:
    from insid3_ext.builder import build_encoder, build_model
    cfgs = [load_config(c) for c in args.configs]
    enc = build_encoder(cfgs[0].model.model_size, cfgs[0].model.weights_path, cfgs[0].model.dinov3_repo)
    ncol = 3 + len(cfgs)          # reference | target | original features | one column per config
    fig, ax = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.4))
    ax[0].imshow(Image.open(args.ref_image).convert("RGB"))
    ax[0].set_title("reference")
    ax[1].imshow(Image.open(args.target).convert("RGB"))
    ax[1].set_title("target")
    for j, cfg in enumerate(cfgs):
        model = build_model(cfg, encoder=enc).to(cfg.model.device).eval()
        f, d, ref_size = _features(model, args.ref_image, args.target)
        if j == 0:                # original (non-debiased) DINOv3 similarity, shown once
            v0 = _ref_vector(f[0, 0], args, model, ref_size)
            sim0 = (v0[:, None, None] * f[0, 1]).sum(dim=0)        # cosine of every target patch with v0
            ax[2].imshow(sim0.float().cpu().numpy(), cmap="jet")
            ax[2].set_title("original features", fontsize=8)
        v = _ref_vector(d[0, 0], args, model, ref_size)
        sim = (v[:, None, None] * d[0, 1]).sum(dim=0).float().cpu().numpy()   # (h, w) cosine map
        ax[j + 3].imshow(sim, cmap="jet")
        ax[j + 3].set_title(f"{cfg.run.exp_name}\n(debias={cfg.debias.mode})", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


def plot_prototypes(args) -> None:
    from insid3_ext.builder import build_model
    from insid3_ext.matching import aggregate_similarities
    from insid3_ext.prototypes import build_reference_prototypes
    from utils.data import downsample_mask, load_mask
    cfg = load_config(args.config)
    model = build_model(cfg).to(cfg.model.device).eval()
    f, d, _ = _features(model, args.ref_image, args.target)
    C, h, w = f.shape[2:]
    mask = load_mask(args.ref_mask, model.image_size, model.device).unsqueeze(0)  # (1,1,H,W)
    group = f[0, :1] if cfg.prototypes.cluster_space == "original" else d[0, :1]
    pset = build_reference_prototypes(d[0, :1], group, mask, cfg.prototypes)
    mk = downsample_mask(mask, h, w)
    assign = torch.full((h, w), -1, device=mk.device)
    toks = d[0, 0][:, mk].T
    assign[mk] = (toks @ pset.prototypes.T).argmax(1)
    C_, h_, w_ = d[0, 1].shape
    sims = (pset.prototypes @ d[0, 1].reshape(C_, -1)).reshape(-1, h_, w_)   # (M, h, w): one map per prototype
    agg = aggregate_similarities(sims.permute(1, 2, 0), pset.weights, cfg.prototypes.token_aggregation
                                 if cfg.prototypes.token_aggregation != "single" else "weighted",
                                 cfg.prototypes.lse_temperature)
    M = pset.num
    fig, ax = plt.subplots(1, M + 2, figsize=(3 * (M + 2), 3.2))
    ax[0].imshow(assign.cpu().numpy(), cmap="tab10", vmin=-1, vmax=9)
    ax[0].set_title(f"reference: {M} prototypes")
    for j in range(M):
        ax[j + 1].imshow(sims[j].float().cpu().numpy(), cmap="jet")
        ax[j + 1].set_title(f"proto {j} (w={float(pset.weights[j]):.2f})")
    ax[M + 1].imshow(agg.float().cpu().numpy(), cmap="jet")
    ax[M + 1].set_title(f"aggregated ({cfg.prototypes.token_aggregation})")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("calibration")
    a.add_argument("--calibration", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("similarity")
    b.add_argument("--configs", nargs="+", required=True)
    b.add_argument("--ref-image", required=True)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref-mask")
    g.add_argument("--ref-point", help="x,y in original reference pixels")
    b.add_argument("--target", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("prototypes")
    c.add_argument("--config", required=True)
    c.add_argument("--ref-image", required=True)
    c.add_argument("--ref-mask", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"calibration": plot_calibration, "similarity": plot_similarity, "prototypes": plot_prototypes}[args.cmd](args)


if __name__ == "__main__":
    main()
