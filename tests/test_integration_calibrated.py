"""Integration: calibration file → calibrated/image/token debiasing → segmentation → plots."""

import copy
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from conftest import IMAGE_SIZE, ROOT, make_image
from insid3_ext.calibration import PositionalStatsAccumulator
from insid3_ext.config import DebiasConfig, PrototypeConfig
from insid3_ext.model import INSID3Ext

KW = dict(image_size=IMAGE_SIZE, svd_components=8, resize_to_orig_size=False, device="cpu")


@pytest.fixture
def calib_file(encoder, tmp_path):
    model = INSID3Ext(copy.deepcopy(encoder), **KW)
    svd = model._pos_svd
    acc = PositionalStatsAccumulator(svd.U, svd.coeff, 8, svd.grid)
    imgs = torch.cat([make_image(box=(8 * i % 64, 8 * i % 48, 8 * i % 64 + 48, 8 * i % 48 + 64), seed=i,
                                 color_box=((i % 3) / 3, 0.5, 1 - (i % 5) / 5)) for i in range(24)])
    feats = model._extract_features(imgs.unsqueeze(0))[0]
    acc.update(F.normalize(feats, dim=1))
    cal = acc.finalize(meta={"model_size": "toy", "image_size": IMAGE_SIZE}, S=svd.S)
    path = tmp_path / "calib.pt"
    torch.save(cal, path)
    return str(path)


@pytest.mark.parametrize("mode", ["calibrated", "image", "token"])
def test_calibrated_modes_run(encoder, episode, calib_file, mode):
    cfg = DebiasConfig(mode=mode, calibration_path=calib_file)
    model = INSID3Ext(copy.deepcopy(encoder), debias_cfg=cfg, model_size="toy",
                      proto_cfg=PrototypeConfig(enabled=True, num_prototypes=2), **KW)
    ref, m, tgt = episode
    out = model.predict_mask(ref, m, tgt)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE) and out.dtype == torch.bool


def test_calibration_metadata_checked(encoder, calib_file):
    with pytest.raises(ValueError, match="model_size"):
        INSID3Ext(copy.deepcopy(encoder), debias_cfg=DebiasConfig(mode="calibrated", calibration_path=calib_file),
                  model_size="large", **KW)


def test_calibration_plots(calib_file, tmp_path):
    out = tmp_path / "figs"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "visualize.py"), "calibration",
                    "--calibration", calib_file, "--out", str(out)], check=True, capture_output=True)
    assert {"calibration_spectra.png", "calibration_tokens.png", "calibration_templates.png"} <= set(os.listdir(out))
