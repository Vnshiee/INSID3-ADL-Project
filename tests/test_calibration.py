"""Task 1: positional-statistics estimation on synthetic data with known ground truth."""

import torch
import torch.nn.functional as F

from insid3_ext.calibration import PositionalStatsAccumulator, summarize_calibration


def _synthetic(n_images=200, C=16, P=36, pos_dims=3, pos_scale=1.0, seed=0):
    """Features = positional field on the first `pos_dims` basis vectors + image content."""
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(C, C, generator=g))[0]
    field = torch.randn(pos_dims, P, generator=g)
    field = field - field.mean(1, keepdim=True)
    coeff = 0.3 * torch.randn(n_images, C, P, generator=g)              # content, all comps
    coeff[:, :pos_dims] += pos_scale * field[None]                       # positional part
    X = torch.einsum("ck,nkp->ncp", U, coeff)
    return U, field, X.reshape(n_images, C, 6, 6), coeff


def test_rho_separates_positional_components():
    U, field, X, _ = _synthetic()
    acc = PositionalStatsAccumulator(U, torch.zeros(16, 36), svd_components=3, grid=(6, 6))
    for i in range(0, 200, 50):
        acc.update(X[i:i + 50])
    cal = acc.finalize(meta={"model_size": "toy", "image_size": 96})
    rho = cal["rho"]
    assert torch.all(rho[:3] > 0.8)          # truth: 1/(1+0.09) ≈ 0.92
    assert torch.all(rho[3:] < 0.05)
    # template recovers the positional field (up to estimation noise)
    cos = F.cosine_similarity(cal["template"][:3], field, dim=1)
    assert torch.all(cos > 0.98)
    s = summarize_calibration(cal, 3)
    assert s["frac_positional_energy_in_top_s"] > 0.95


def test_bias_correction_no_position():
    """Without positional signal the corrected positional variance must be ≈ 0."""
    U, _, X, _ = _synthetic(n_images=30, pos_scale=0.0)
    acc = PositionalStatsAccumulator(U, torch.zeros(16, 36), svd_components=3, grid=(6, 6))
    acc.update(X)
    rho = acc.finalize()["rho"]
    assert float(rho.max()) < 0.05


def test_grid_check():
    U, _, X, _ = _synthetic(n_images=4)
    acc = PositionalStatsAccumulator(U, torch.zeros(16, 36), 3, (5, 5))
    try:
        acc.update(X)
    except ValueError:
        return
    raise AssertionError("grid mismatch not detected")


def test_shapes_of_outputs():
    U, _, X, _ = _synthetic(n_images=10)
    acc = PositionalStatsAccumulator(U, torch.randn(16, 36), 3, (6, 6))
    acc.update(X)
    cal = acc.finalize()
    assert cal["per_image_second_moment"].shape == (10, 16)
    assert cal["per_image_noise_corr"].shape == (10, 16)
    assert cal["token_energy_mean"].shape == (36,)
    assert torch.all((cal["rho"] >= 0) & (cal["rho"] <= 1))
