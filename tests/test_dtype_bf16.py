"""Dtype robustness of the NEW modules with bf16 inputs.

Finding (recorded in docs/REPOSITORY_AUDIT.md §5): the *official* pipeline cannot run on bf16
features — ``utils.clustering.agglomerative_clustering`` calls ``.numpy()``, which rejects bf16.
Because the official code runs on CUDA under bf16 autocast, the DINOv3 output it receives must be
fp32 (the final LayerNorm runs in fp32 under autocast). Hence the full pipeline is not tested in
bf16; only our modules are, so they stay safe if a backbone ever returns reduced precision.
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import IMAGE_SIZE
from insid3_ext.config import DebiasConfig, PrototypeConfig
from insid3_ext.debiasing import PositionalDebiaser, estimate_positional_svd
from insid3_ext.matching import aggregate_similarities, region_scores_chamfer, region_scores_ot
from insid3_ext.prototypes import build_reference_prototypes


def test_official_clustering_rejects_bf16():
    from utils.clustering import agglomerative_clustering
    with pytest.raises(TypeError):
        agglomerative_clustering(F.normalize(torch.randn(10, 4), dim=1).bfloat16(), 0.6)


@pytest.mark.parametrize("cfg", [DebiasConfig(mode="hard"), DebiasConfig(mode="weighted", alpha=0.5),
                                 DebiasConfig(mode="spectral", knee_rank=8),
                                 DebiasConfig(mode="image", variance_source="noise"),
                                 DebiasConfig(mode="token", template_source="noise")])
def test_debiaser_bf16(encoder, cfg):
    svd = estimate_positional_svd(encoder, IMAGE_SIZE, "cpu")
    x = F.normalize(torch.randn(1, 2, 32, 8, 8), dim=2).bfloat16()
    out = PositionalDebiaser.from_config(cfg, svd, 8, None)(x)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    assert torch.allclose(out.float().norm(dim=2), torch.ones(1, 2, 8, 8), atol=2e-2)


def test_prototypes_and_matching_bf16():
    f = F.normalize(torch.randn(1, 32, 8, 8), dim=1).bfloat16()
    m = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    m[..., :64, :64] = True
    ps = build_reference_prototypes(f, f, m, PrototypeConfig(num_prototypes=3))
    assert ps.prototypes.dtype == torch.float32 and torch.isfinite(ps.prototypes).all()
    sims = F.normalize(torch.randn(64, 32), dim=1) @ ps.prototypes.T
    labels = torch.arange(64) % 5
    for mode in ("max", "weighted", "lse"):
        assert torch.isfinite(aggregate_similarities(sims, ps.weights, mode)).all()
    assert torch.isfinite(region_scores_chamfer(sims, labels, 5, ps.weights)).all()
    assert torch.isfinite(region_scores_ot(sims.bfloat16(), labels, 5, ps.weights)).all()
