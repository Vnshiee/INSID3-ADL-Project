"""Task 1: positional projection, adaptive suppression, normalisation, shapes."""

import pytest
import torch
import torch.nn.functional as F

from conftest import IMAGE_SIZE
from insid3_ext.config import DebiasConfig
from insid3_ext.debiasing import (PositionalDebiaser, calibrated_gains, estimate_positional_svd,
                                  spectral_gains, subspace_agreement, weighted_gains)

S_COMP = 8


@pytest.fixture
def svd(encoder):
    return estimate_positional_svd(encoder, IMAGE_SIZE, "cpu")


@pytest.fixture
def feats():
    g = torch.Generator().manual_seed(0)
    return F.normalize(torch.randn(1, 3, 32, 8, 8, generator=g), dim=2)   # (B, T, C, h, w)


def _deb(svd, **kw):
    return PositionalDebiaser.from_config(DebiasConfig(**kw), svd, S_COMP, None)


def test_svd_properties(svd):
    U, S = svd.U, svd.S
    assert U.shape == (32, 32) and S.shape == (32,)
    assert torch.allclose(U.T @ U, torch.eye(32), atol=1e-5)
    assert torch.all(S[:-1] >= S[1:])
    assert svd.coeff.shape == (32, 64)
    assert torch.allclose(svd.coeff.mean(1), torch.zeros(32), atol=1e-5)  # centred tokens


def test_hard_projection_is_orthogonal_complement(svd, feats):
    out = _deb(svd, mode="hard")(feats)
    B = svd.U[:, :S_COMP]
    X = out.reshape(3, 32, 64)
    assert torch.allclose(torch.einsum("cs,ncp->nsp", B, X), torch.zeros(3, S_COMP, 64), atol=1e-5)
    assert torch.allclose(out.norm(dim=2), torch.ones(1, 3, 8, 8), atol=1e-5)


def test_projection_idempotent(svd):
    B = svd.U[:, :S_COMP]
    P = torch.eye(32) - B @ B.T
    assert torch.allclose(P @ P, P, atol=1e-5)


@pytest.mark.parametrize("mode,kw", [("none", {}), ("weighted", {"alpha": 0.0}),
                                     ("calibrated_zero", {})])
def test_zero_suppression_is_identity(svd, feats, mode, kw):
    if mode == "calibrated_zero":
        d = PositionalDebiaser("calibrated", svd.U, gains=torch.zeros(32))
    else:
        d = _deb(svd, mode=mode, **kw)
    assert torch.allclose(d(feats), feats, atol=1e-6)


def test_weighted_alpha1_equals_hard(svd, feats):
    assert torch.allclose(_deb(svd, mode="weighted", alpha=1.0)(feats), _deb(svd, mode="hard")(feats), atol=1e-5)


def test_weighted_monotone_in_alpha(svd, feats):
    B = svd.U[:, :S_COMP]
    energies = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        X = feats.reshape(3, 32, 64)
        Xd = X - a * B @ (B.T @ X)                    # before re-normalisation
        energies.append(float((B.T @ Xd).pow(2).sum()))
    assert all(e1 >= e2 for e1, e2 in zip(energies, energies[1:]))


def test_spectral_gains_limits(svd):
    knee = 3
    lam = spectral_gains(svd.S, 32, knee_rank=knee, sharpness=1.0)
    assert lam.shape == (32,) and torch.all((lam >= 0) & (lam <= 1))
    assert torch.all(lam[:-1] >= lam[1:] - 1e-7)          # monotone in k
    assert abs(float(lam[knee - 1]) - 0.5) < 1e-5         # 0.5 at the knee
    hard_like = spectral_gains(svd.S, 32, knee_rank=knee, sharpness=200.0)
    assert torch.all(hard_like[:knee - 1] > 0.99) and torch.all(hard_like[knee:] < 0.01)


def test_spectral_rejects_bad_knee(svd):
    with pytest.raises(ValueError):
        spectral_gains(svd.S, 32, knee_rank=100, sharpness=1.0)


def test_calibrated_gains_clamped():
    rho = torch.tensor([0.9, 0.5, 0.1, 0.0])
    assert torch.allclose(calibrated_gains(rho, 4, 1.0), rho)
    assert torch.allclose(calibrated_gains(rho, 4, 2.0), torch.tensor([1.0, 1.0, 0.2, 0.0]))
    assert torch.allclose(weighted_gains(3, 0.3), torch.full((3,), 0.3))


def test_calibrated_indicator_equals_hard(svd, feats):
    rho = torch.zeros(32)
    rho[:S_COMP] = 1.0
    d = PositionalDebiaser("calibrated", svd.U, gains=calibrated_gains(rho, 32, 1.0))
    assert torch.allclose(d(feats), _deb(svd, mode="hard")(feats), atol=1e-5)


def test_image_mode_gain_range_and_identity(svd, feats):
    d = _deb(svd, mode="image", variance_source="noise", strength=1.0)
    out = d(feats)
    assert out.shape == feats.shape
    assert 0.0 <= d.last_stats["mean_lambda"] <= 1.0
    d0 = _deb(svd, mode="image", variance_source="noise", strength=0.0)
    assert torch.allclose(d0(feats), feats, atol=1e-6)


def test_image_mode_is_image_dependent(svd):
    """Images with more energy along u_k get a smaller λ_k (Wiener behaviour)."""
    d = _deb(svd, mode="image", variance_source="noise")
    u = svd.U[:, 0]
    base = F.normalize(torch.randn(1, 1, 32, 8, 8), dim=2)
    strong = F.normalize(base + 3 * u.view(1, 1, 32, 1, 1), dim=2)
    d(base)
    lam_base = d.last_stats["mean_lambda"]
    d(torch.cat([strong], 1))
    assert d.last_stats["mean_lambda"] != lam_base


def test_token_mode_noise_template_flattens_constant_image(svd, encoder):
    """On the constant image itself, removing the full template leaves identical tokens."""
    from torchvision.transforms.functional import normalize
    img = normalize(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    f = F.normalize(encoder.get_intermediate_layers(img)[0], dim=1)[None]     # (1,1,C,h,w)
    d = _deb(svd, mode="token", template_source="noise", token_gain="fixed", strength=1.0)
    out = d(f).reshape(32, 64)
    assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-4)


def test_token_mode_ls_gain_bounds(svd, feats):
    d = _deb(svd, mode="token", template_source="noise", token_gain="least_squares", max_gain=1.5)
    out = d(feats)
    assert out.shape == feats.shape
    assert 0.0 <= d.last_stats["mean_gamma"] <= 1.5
    d0 = _deb(svd, mode="token", template_source="noise", strength=0.0)
    assert torch.allclose(d0(feats), feats, atol=1e-6)


def test_token_mode_grid_mismatch_raises(svd):
    d = _deb(svd, mode="token", template_source="noise")
    with pytest.raises(ValueError):
        d(F.normalize(torch.randn(1, 1, 32, 4, 4), dim=2))


@pytest.mark.parametrize("mode", ["none", "hard", "weighted", "image", "token"])
def test_output_shape_dtype_and_unit_norm(svd, feats, mode):
    d = _deb(svd, mode=mode, variance_source="noise", template_source="noise", alpha=0.5)
    out = d(feats)
    assert out.shape == feats.shape and out.dtype == feats.dtype
    assert torch.allclose(out.norm(dim=2), torch.ones(1, 3, 8, 8), atol=1e-4)


def test_requires_calibration(svd):
    with pytest.raises(ValueError):
        _deb(svd, mode="calibrated")


def test_subspace_agreement(svd):
    assert abs(subspace_agreement(svd.U, svd.U, 5) - 1.0) < 1e-6
    flipped = svd.U.clone()
    flipped[:, 0] *= -1                                         # sign ambiguity is irrelevant
    assert abs(subspace_agreement(svd.U, flipped, 5) - 1.0) < 1e-6
