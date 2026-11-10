"""The extended model must reproduce the official INSID3 when the extensions are off.

Compares ``models.insid3.INSID3`` (official, unmodified) with ``INSID3Ext`` on CPU with
the fake encoder: positional basis, debiased features, predicted masks, keypoints.
"""

import copy

import pytest
import torch

from conftest import IMAGE_SIZE, make_image, make_mask
from insid3_ext.config import DebiasConfig, PrototypeConfig
from insid3_ext.model import INSID3Ext
from models.insid3 import INSID3

KW = dict(image_size=IMAGE_SIZE, svd_components=8, tau=0.6, merge_threshold=0.2,
          mask_refiner="bilinear", resize_to_orig_size=False, device="cpu")


def _pair(encoder, debias=None, proto=None):
    orig = INSID3(copy.deepcopy(encoder), **KW).eval()
    ext = INSID3Ext(copy.deepcopy(encoder), debias_cfg=debias, proto_cfg=proto, **KW).eval()
    return orig, ext


def test_positional_basis_identical(encoder):
    orig, ext = _pair(encoder)
    assert torch.equal(orig.positional_basis, ext.positional_basis)
    assert ext._pos_svd.U.shape[1] >= KW["svd_components"]


def test_hard_debias_identical(encoder, episode):
    orig, ext = _pair(encoder)
    ref, _, tgt = episode
    f = torch.nn.functional.normalize(orig._extract_features(torch.cat([ref, tgt]).unsqueeze(0)), dim=2)
    assert torch.equal(orig._debias_features(f), ext._debias_features(f))
    # the debiaser's own 'hard' path (used outside the model) is the same operation
    assert torch.allclose(orig._debias_features(f), ext.debiaser(f), atol=0, rtol=0)


def test_predict_mask_identical_extensions_off(encoder, episode):
    orig, ext = _pair(encoder)
    ref, ref_mask, tgt = episode
    a = orig.predict_mask(ref, ref_mask, tgt)
    b = ext.predict_mask(ref, ref_mask, tgt)
    assert a.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_task2_path_m1_reproduces_baseline_1shot(encoder, seed):
    orig, ext = _pair(encoder, proto=PrototypeConfig(enabled=True, num_prototypes=1,
                                                     token_aggregation="max", region_matching="prototype"))
    ref = make_image(box=(16, 16, 64, 64), seed=10 + seed)
    tgt = make_image(box=(40 + 4 * seed, 40, 104, 104), seed=20 + seed)
    m = make_mask((16, 16, 64, 64)).unsqueeze(0)
    assert torch.equal(orig.predict_mask(ref, m, tgt), ext.predict_mask(ref, m, tgt))


def test_task2_single_aggregation_reproduces_baseline_multishot(encoder):
    orig, ext = _pair(encoder, proto=PrototypeConfig(enabled=True, num_prototypes=3,
                                                     token_aggregation="single", region_matching="prototype"))
    refs = torch.cat([make_image(box=(16, 16, 64, 64), seed=1), make_image(box=(48, 40, 100, 100), seed=3)])
    masks = torch.stack([make_mask((16, 16, 64, 64)), make_mask((48, 40, 100, 100))])
    tgt = make_image(box=(56, 48, 112, 112), seed=2)
    assert torch.equal(orig.predict_mask(refs, masks, tgt), ext.predict_mask(refs, masks, tgt))


def test_weighted_alpha1_matches_hard_features(encoder, episode):
    orig, ext = _pair(encoder, debias=DebiasConfig(mode="weighted", alpha=1.0))
    ref, _, tgt = episode
    f = torch.nn.functional.normalize(orig._extract_features(torch.cat([ref, tgt]).unsqueeze(0)), dim=2)
    assert torch.allclose(orig._debias_features(f), ext._debias_features(f), atol=1e-5)


def test_keypoint_matching_identical(encoder, episode):
    orig, ext = _pair(encoder)
    ref, _, tgt = episode
    kps = torch.tensor([[40.0, 40.0], [20.0, 60.0]])
    for deb in (False, True):
        orig._orig_ref_size = ext._orig_ref_size = (IMAGE_SIZE, IMAGE_SIZE)
        orig._orig_tgt_size = ext._orig_tgt_size = (IMAGE_SIZE, IMAGE_SIZE)
        a = orig.predict_keypoint(ref, kps, tgt, use_debiased=deb)
        b = ext.predict_keypoint(ref, kps, tgt, use_debiased=deb)
        assert torch.equal(a, b)


def test_public_api_segment(encoder, episode):
    """set_reference / set_target / segment with tensors (the evaluation code path)."""
    _, ext = _pair(encoder, proto=PrototypeConfig(enabled=True, num_prototypes=3))
    ref, ref_mask, tgt = episode
    ext.set_reference(ref, ref_mask[0])
    ext.set_target(tgt)
    out = ext.segment()
    assert out.dtype == torch.bool and out.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert ext.last_diagnostics["num_prototypes"] >= 1
