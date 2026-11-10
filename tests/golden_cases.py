"""Golden outputs: proof that the readability rewrite changed no number.

`compute_cases()` runs every debiasing mode and every prototype option of `insid3_ext` on fixed toy inputs
(GoldenEncoder below, CPU) and returns {case name: tensor}. The outputs of the code *before* the rewrite (v5)
are stored in tests/golden/golden_v5_fullrank.pt (created with `python tests/golden_cases.py` on the v5 code);
tests/test_golden.py checks that the current code reproduces every one of them.
"""

import copy
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from conftest import IMAGE_SIZE, FakeEncoder, make_image, make_mask  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "golden_v5_fullrank.pt")
KW = dict(image_size=IMAGE_SIZE, svd_components=8, tau=0.6, merge_threshold=0.2,
          mask_refiner="bilinear", resize_to_orig_size=False, device="cpu")


class GoldenEncoder(FakeEncoder):
    """FakeEncoder with its positional field replaced by a FULL-RANK one with well separated strengths.

    Why: FakeEncoder's empty-image features have rank 3 (constant image + a 3-direction field), so directions
    4..C of their SVD are an arbitrary basis of a null space, which different CPUs / LAPACK libraries choose
    differently, so the golden values would depend on the machine. Here every direction has its own strength
    (ratio 0.88 between neighbours), so the SVD is unique up to sign on every machine.
    """

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(4321)
        C, h = self.pos.shape[1], self.pos.shape[-1]
        dirs = torch.linalg.qr(torch.randn(C, C, generator=g, dtype=torch.float64))[0]         # (C, C)
        patterns = torch.linalg.qr(torch.randn(h * h, C, generator=g, dtype=torch.float64))[0]  # (P, C)
        strength = 1.5 * 0.88 ** torch.arange(C, dtype=torch.float64)
        field = (dirs * strength) @ patterns.T                                                  # (C, P)
        self.pos = field.float().view(1, C, h, h)


def _episode(shots):
    boxes = [(16, 16, 64, 64), (48, 40, 100, 100), (8, 60, 70, 120)]
    refs = torch.cat([make_image(box=boxes[i], seed=40 + i) for i in range(shots)])
    masks = torch.stack([make_mask(boxes[i]) for i in range(shots)])
    return refs, masks, make_image(box=(56, 48, 112, 112), seed=2)


def _calibration(encoder, path):
    from insid3_ext.calibration import PositionalStatsAccumulator
    from insid3_ext.debiasing import estimate_positional_svd
    svd = estimate_positional_svd(encoder, IMAGE_SIZE, "cpu")
    acc = PositionalStatsAccumulator(svd.U, svd.coeff, 8, svd.grid)
    imgs = torch.cat([make_image(box=(4 * i, 8, 4 * i + 60, 90), seed=i) for i in range(12)])
    acc.update(F.normalize(encoder.get_intermediate_layers(imgs)[0], dim=1))
    cal = acc.finalize(meta={"model_size": "toy", "image_size": IMAGE_SIZE}, S=svd.S)
    torch.save(cal, path)
    return cal


def _diag(model):
    """All numeric diagnostics of the last prediction (seed score, cross score, counts, score scale)."""
    d = model.last_diagnostics
    return torch.tensor([float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))], dtype=torch.float64)


@torch.no_grad()
def compute_cases(tmpdir):
    from insid3_ext.config import DebiasConfig, PrototypeConfig
    from insid3_ext.matching import (aggregate_similarities, region_scores_chamfer, region_scores_ot,
                                     unbalanced_sinkhorn_plan)
    from insid3_ext.model import INSID3Ext
    from insid3_ext.prototypes import build_reference_prototypes, spherical_kmeans

    torch.manual_seed(0)
    enc = GoldenEncoder().eval()
    cpath = os.path.join(tmpdir, "cal.pt")
    cal = _calibration(enc, cpath)
    out = {}
    for k in ("rho", "pos_second_moment", "template", "second_moment", "feat_mean", "feat_cov",
              "token_energy_mean", "per_image_noise_corr"):
        out[f"cal/{k}"] = cal[k].float()

    feats = F.normalize(torch.randn(1, 3, 32, 8, 8, generator=torch.Generator().manual_seed(5)), dim=2)
    debias_cases = {
        "none": dict(mode="none"), "hard": dict(mode="hard"), "weighted": dict(mode="weighted", alpha=0.5),
        "spectral_g1": dict(mode="spectral", knee_rank=8, sharpness=1.0),
        "spectral_g4": dict(mode="spectral", knee_rank=8, sharpness=4.0),
        "calibrated": dict(mode="calibrated"), "calibrated_b2": dict(mode="calibrated", strength=2.0),
        "image_cal": dict(mode="image"), "image_noise": dict(mode="image", variance_source="noise"),
        "token_ls": dict(mode="token"), "token_fixed": dict(mode="token", token_gain="fixed"),
        "token_noise": dict(mode="token", template_source="noise"),
        "random": dict(mode="random"), "pca": dict(mode="pca"), "rho": dict(mode="rho_threshold"),
        "center_hard": dict(mode="hard", center=True), "center_none": dict(mode="none", center=True),
    }
    ref1, m1, tgt = _episode(1)
    ref3, m3, _ = _episode(3)
    # Keypoints chosen so that, in EVERY mode, the best target patch beats the second best by > 1.8e-3.
    # (Keypoint prediction starts with an argmax; at a near-tie, last-digit differences between machines
    # would pick the other patch and move the point by 16 px — an input artefact, not a code change.)
    kps = torch.tensor([[104.0, 72.0], [8.0, 120.0], [24.0, 120.0]])
    for name, kw in debias_cases.items():
        model = INSID3Ext(copy.deepcopy(enc), debias_cfg=DebiasConfig(calibration_path=cpath, **kw),
                          model_size="toy", full_diagnostics=True, **KW).eval()
        out[f"deb/{name}/feat"] = model._debias_features(feats)
        out[f"deb/{name}/mask1"] = model.predict_mask(ref1, m1, tgt)
        out[f"deb/{name}/diag1"] = _diag(model)
        out[f"deb/{name}/mask3"] = model.predict_mask(ref3, m3, tgt)
        out[f"deb/{name}/diag3"] = _diag(model)
        model._orig_ref_size = model._orig_tgt_size = (IMAGE_SIZE, IMAGE_SIZE)
        out[f"deb/{name}/kp"] = model.predict_keypoint(ref1, kps, tgt, use_debiased=True)

    proto_cases = {
        "single": dict(token_aggregation="single", num_prototypes=3),
        "max_m1": dict(num_prototypes=1), "max_m3": dict(num_prototypes=3),
        "weighted_m3": dict(num_prototypes=3, token_aggregation="weighted"),
        "lse_m3": dict(num_prototypes=3, token_aggregation="lse"),
        "chamfer": dict(num_prototypes=3, region_matching="chamfer"),
        "ot": dict(num_prototypes=3, token_aggregation="lse", region_matching="ot"),
        "agglo": dict(num_prototypes=3, clustering="agglomerative", ref_tau=0.9),
        "global": dict(num_prototypes=3, include_global=True),
        "minfrac": dict(num_prototypes=4, min_cluster_fraction=0.2),
        "groupdeb": dict(num_prototypes=3, cluster_space="debiased"),
        "scal": dict(num_prototypes=3, token_aggregation="lse", score_calibration="reference"),
        "pooled": dict(num_prototypes=3, multishot="pooled", token_aggregation="lse"),
        "pooled_m1": dict(num_prototypes=1, multishot="pooled"),
        "adaptive": dict(num_prototypes=5, min_points_per_prototype=100),
    }
    for name, kw in proto_cases.items():
        model = INSID3Ext(copy.deepcopy(enc), proto_cfg=PrototypeConfig(enabled=True, **kw), **KW).eval()
        out[f"proto/{name}/mask1"] = model.predict_mask(ref1, m1, tgt)
        out[f"proto/{name}/diag1"] = _diag(model)
        out[f"proto/{name}/mask3"] = model.predict_mask(ref3, m3, tgt)
        out[f"proto/{name}/diag3"] = _diag(model)
    diag = INSID3Ext(copy.deepcopy(enc), full_diagnostics=True, **KW).eval()
    out["diag/mask3"] = diag.predict_mask(ref3, m3, tgt)
    out["diag/values"] = _diag(diag)

    # low-level functions
    g = torch.Generator().manual_seed(3)
    X = F.normalize(torch.randn(80, 16, generator=g), dim=1)
    lab, cen, obj = spherical_kmeans(X, 4, 30, torch.Generator().manual_seed(1))
    out["km/labels"], out["km/centers"], out["km/obj"] = lab, cen, torch.tensor(obj)
    labw, _, _ = spherical_kmeans(X, 4, 30, torch.Generator().manual_seed(1), weights=torch.rand(80, generator=g))
    out["km/labels_w"] = labw
    f = F.normalize(torch.randn(2, 16, 8, 8, generator=g), dim=1)
    mm = torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    mm[0, :, :64, :64] = mm[1, :, 32:, 32:] = True
    for ms in ("per_shot", "pooled"):
        ps = build_reference_prototypes(f, f, mm, PrototypeConfig(num_prototypes=3, multishot=ms))
        out[f"ps/{ms}/q"], out[f"ps/{ms}/w"] = ps.prototypes, ps.weights
    sims = torch.rand(40, 4, generator=g) * 2 - 1
    labels = torch.arange(40) % 5
    w = torch.tensor([0.4, 0.3, 0.2, 0.1])
    for mode in ("max", "weighted", "lse"):
        out[f"agg/{mode}"] = aggregate_similarities(sims, w, mode, 0.05)
    out["region/chamfer"] = region_scores_chamfer(sims, labels, 5, w)
    out["region/ot"] = region_scores_ot(sims, labels, 5, w, 0.05, 0.5, 50)
    out["region/plan"] = unbalanced_sinkhorn_plan(sims, labels, 5, w, 0.1, 1.0, 30)
    return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        res = compute_cases(d)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    torch.save(res, GOLDEN)
    print(f"saved {len(res)} golden outputs to {GOLDEN}")
