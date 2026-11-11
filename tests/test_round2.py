"""Tests for round 2: control debias modes, pooled multi-shot, diagnostics path,
score calibration, covariance calibration, correspondence bootstrap, positional-bias analysis."""

import copy
import json
import os
import runpy
import sys

import numpy as np
import pytest
import torch
import yaml
import torch.nn.functional as F
from PIL import Image

from conftest import IMAGE_SIZE, ROOT, FakeEncoder, make_image, make_mask
from insid3_ext.calibration import PositionalStatsAccumulator
from insid3_ext.config import ConfigError, DebiasConfig, PrototypeConfig, load_config
from insid3_ext.debiasing import PositionalDebiaser, estimate_positional_svd
from insid3_ext.model import INSID3Ext
from insid3_ext.prototypes import build_reference_prototypes, spherical_kmeans
from models.insid3 import INSID3

KW = dict(image_size=IMAGE_SIZE, svd_components=8, tau=0.6, merge_threshold=0.2,
          mask_refiner="bilinear", resize_to_orig_size=False, device="cpu")
SCRIPTS = os.path.join(ROOT, "scripts")


def _shots(n):
    boxes = [(16, 16, 64, 64), (48, 40, 100, 100), (8, 60, 70, 120), (30, 10, 90, 70), (60, 60, 120, 120)]
    refs = torch.cat([make_image(box=boxes[i], seed=40 + i) for i in range(n)])
    masks = torch.stack([make_mask(boxes[i]) for i in range(n)])
    return refs, masks, make_image(box=(56, 48, 112, 112), seed=2)


@pytest.fixture
def calib(encoder):
    svd = estimate_positional_svd(encoder, IMAGE_SIZE, "cpu")
    acc = PositionalStatsAccumulator(svd.U, svd.coeff, 8, svd.grid)
    imgs = torch.cat([make_image(box=(4 * i, 8, 4 * i + 60, 90), seed=i) for i in range(12)])
    acc.update(F.normalize(encoder.get_intermediate_layers(imgs)[0], dim=1))
    return acc.finalize(meta={"model_size": "toy", "image_size": IMAGE_SIZE}, S=svd.S), svd


# ───────────── control debias modes ─────────────

def test_covariance_in_calibration(calib):
    cal, _ = calib
    cov = cal["feat_cov"]
    assert cov.shape == (32, 32) and torch.allclose(cov, cov.T, atol=1e-5)
    assert torch.linalg.eigvalsh(cov.double()).min() > -1e-5


@pytest.mark.parametrize("mode", ["random", "pca"])
def test_control_modes_are_rank_s_projections(calib, mode):
    cal, svd = calib
    d = PositionalDebiaser.from_config(DebiasConfig(mode=mode, calibration_path="x"), svd, 8, cal)
    U = d.U
    assert U.shape == (32, 8)
    assert torch.allclose(U.T @ U, torch.eye(8), atol=1e-4)
    x = F.normalize(torch.randn(1, 2, 32, 8, 8), dim=2)
    out = d(x).reshape(2, 32, 64)
    assert torch.allclose(U.T @ out, torch.zeros(2, 8, 64), atol=1e-4)      # subspace removed


def test_random_mode_deterministic_and_differs_from_svd(calib):
    cal, svd = calib
    a = PositionalDebiaser.from_config(DebiasConfig(mode="random", control_seed=3), svd, 8, None).U
    b = PositionalDebiaser.from_config(DebiasConfig(mode="random", control_seed=3), svd, 8, None).U
    assert torch.equal(a, b)
    from insid3_ext.debiasing import subspace_agreement
    assert subspace_agreement(a, svd.U, 8) < 0.9


def test_pca_mode_requires_covariance(calib):
    cal, svd = calib
    cal = {k: v for k, v in cal.items() if k != "feat_cov"}
    with pytest.raises(ValueError, match="feat_cov"):
        PositionalDebiaser.from_config(DebiasConfig(mode="pca", calibration_path="x"), svd, 8, cal)


def test_rho_threshold_indicator(calib):
    cal, svd = calib
    cal = dict(cal)
    rho = torch.zeros(32)
    rho[:5] = 0.9
    cal["rho"], cal["U"] = rho, svd.U
    d = PositionalDebiaser.from_config(DebiasConfig(mode="rho_threshold", rho_threshold=0.5,
                                                    calibration_path="x"), svd, 8, cal)
    assert torch.equal(d.gains[:5], torch.ones(5)) and d.gains[5:].sum() == 0
    hard5 = PositionalDebiaser.from_config(DebiasConfig(mode="hard"), svd, 5, None)
    x = F.normalize(torch.randn(1, 2, 32, 8, 8), dim=2)
    assert torch.allclose(d(x), hard5(x), atol=1e-5)


def test_normalize_false_returns_pre_norm(calib):
    _, svd = calib
    x = F.normalize(torch.randn(1, 2, 32, 8, 8), dim=2)
    for mode in ("hard", "weighted"):
        d = PositionalDebiaser.from_config(DebiasConfig(mode=mode, alpha=0.5), svd, 8, None)
        raw = d(x, normalize=False)
        assert torch.all(raw.norm(dim=2) <= 1 + 1e-5)
        assert torch.allclose(F.normalize(raw, dim=2), d(x), atol=1e-6)


def test_new_config_validation():
    base = os.path.join(ROOT, "configs", "baseline.yaml")
    for bad in (["prototypes.multishot=mixed"], ["prototypes.score_calibration=z"], ["debias.rho_threshold=2"],
                ["debias.mode=pca", "debias.calibration_path=null"]):
        with pytest.raises(ConfigError):
            load_config(base, bad)
    assert load_config(base, ["debias.mode=random"]).debias.mode == "random"


# ───────────── pooled multi-shot ─────────────

def test_weighted_kmeans_uniform_equals_unweighted():
    X = F.normalize(torch.randn(60, 8, generator=torch.Generator().manual_seed(0)), dim=1)
    a = spherical_kmeans(X, 3, 30, torch.Generator().manual_seed(1))[0]
    b = spherical_kmeans(X, 3, 30, torch.Generator().manual_seed(1), weights=torch.ones(60))[0]
    assert torch.equal(a, b)


def test_pooled_m1_prototype_equals_original_global():
    f = F.normalize(torch.randn(3, 16, 8, 8), dim=1)
    m = torch.zeros(3, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    m[0, :, :64, :64] = m[1, :, 32:, 32:] = m[2, :, 16:48, :] = True
    ps = build_reference_prototypes(f, f, m, PrototypeConfig(num_prototypes=1, multishot="pooled"))
    assert ps.num == 1 and torch.allclose(ps.prototypes[0], ps.global_prototype, atol=1e-5)


def test_pooled_weights_and_shapes():
    f = F.normalize(torch.randn(2, 16, 8, 8), dim=1)
    m = torch.zeros(2, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    m[0, :, :64, :64] = m[1, :, :, :] = True
    ps = build_reference_prototypes(f, f, m, PrototypeConfig(num_prototypes=4, multishot="pooled"))
    assert 1 <= ps.num <= 4 and abs(float(ps.weights.sum()) - 1) < 1e-5
    assert torch.all(ps.shot_index == -2)


@pytest.mark.parametrize("shots", [2, 5])
def test_pooled_m1_pipeline_reproduces_official_multishot(encoder, shots):
    refs, masks, tgt = _shots(shots)
    orig = INSID3(copy.deepcopy(encoder), **KW).eval()
    ext = INSID3Ext(copy.deepcopy(encoder), proto_cfg=PrototypeConfig(
        enabled=True, num_prototypes=1, multishot="pooled", token_aggregation="max"), **KW).eval()
    assert torch.equal(orig.predict_mask(refs, masks, tgt), ext.predict_mask(refs, masks, tgt))


# ───────────── diagnostics path and score calibration ─────────────

@pytest.mark.parametrize("shots", [1, 3])
def test_full_diagnostics_path_is_identical(encoder, shots):
    refs, masks, tgt = _shots(shots)
    orig = INSID3(copy.deepcopy(encoder), **KW).eval()
    ext = INSID3Ext(copy.deepcopy(encoder), full_diagnostics=True, **KW).eval()
    assert torch.equal(orig.predict_mask(refs, masks, tgt), ext.predict_mask(refs, masks, tgt))
    d = ext.last_diagnostics
    for k in ("forward_tokens", "backward_tokens", "candidate_tokens", "seed_cluster", "selected_clusters"):
        assert k in d
    assert d["score_scale"] == 1.0


def test_score_calibration(encoder, episode):
    ref, m, tgt = episode
    for M, expect_one in ((1, True), (3, False)):
        ext = INSID3Ext(copy.deepcopy(encoder), proto_cfg=PrototypeConfig(
            enabled=True, num_prototypes=M, token_aggregation="lse", score_calibration="reference"), **KW)
        out = ext.predict_mask(ref, m, tgt)
        k = ext.last_diagnostics["score_scale"]
        assert out.shape == (IMAGE_SIZE, IMAGE_SIZE) and k > 0
        if expect_one:
            assert abs(k - 1.0) < 1e-4


# ───────────── scripts ─────────────

def _write_corr(d, dists, exp, mode, debiased, center=False, eval_basis=False):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.yaml"), "w") as fp:
        yaml.safe_dump({"debias": {"mode": mode, "center": center}, "model": {"basis_eval_mode": eval_basis}}, fp)
    with open(os.path.join(d, "episodes.jsonl"), "w") as fp:
        for i, dist in enumerate(dists):
            fp.write(json.dumps({"idx": i, "category": "cat", "pck_thresh": 100.0, "dist": dist}) + "\n")
    with open(os.path.join(d, "metrics.json"), "w") as fp:
        json.dump({"task": "correspondence", "exp_name": exp, "complete": True, "model_size": "base",
                   "image_size": 768, "svd_components": 20, "debias_mode": mode, "debiased": debiased}, fp)


def test_compare_all_correspondence(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 30, size=(200, 5))
    _write_corr(tmp_path / "corr_hard" / "spair" / "test_seed0", base.tolist(), "corr_hard", "hard", True)
    _write_corr(tmp_path / "corr_tok" / "spair" / "test_seed0", (base * 0.8).tolist(), "corr_tok", "token", True)
    # Also mode 'hard', but NOT the reference (centring / eval-mode basis / no debiasing):
    _write_corr(tmp_path / "corr_a_center" / "spair" / "test_seed0", (base * 2).tolist(), "corr_a_center", "hard",
                True, center=True)
    _write_corr(tmp_path / "corr_a_evalbasis" / "spair" / "test_seed0", (base * 2).tolist(), "corr_a_evalbasis",
                "hard", True, eval_basis=True)
    _write_corr(tmp_path / "corr_a_orig" / "spair" / "test_seed0", (base * 2).tolist(), "corr_a_orig", "hard", False)
    monkeypatch.syspath_prepend(SCRIPTS)
    out = str(tmp_path / "cmp")
    monkeypatch.setattr(sys, "argv", ["compare_all.py", "--root", str(tmp_path), "--out", out, "--bootstrap", "200"])
    runpy.run_path(os.path.join(SCRIPTS, "compare_all.py"), run_name="__main__")
    rows = [r for r in json.load(open(out + ".json")) if r.get("task") == "correspondence"]
    assert len(rows) == 16 and all(r["reference"] == "corr_hard" for r in rows)
    rows = [r for r in rows if r["exp"] == "corr_tok"]
    assert len(rows) == 4 and all(r["delta_pck"] >= 0 for r in rows)
    r10 = [r for r in rows if r["alpha"] == 0.1][0]
    assert r10["ci95"][0] <= r10["delta_pck"] <= r10["ci95"][1]


def test_analyze_positional_bias_cli(tmp_path, monkeypatch, calib):
    cal, _ = calib
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(imgs / f"{i}.jpg")
    cal = dict(cal, image_paths=[str(imgs / "0.jpg")], meta=dict(cal["meta"], model_size="large"))
    cpath = str(tmp_path / "cal.pt")
    torch.save(cal, cpath)
    import insid3_ext.builder as builder
    monkeypatch.setattr(builder, "build_encoder", lambda *a, **k: FakeEncoder().eval())
    monkeypatch.syspath_prepend(SCRIPTS)
    out = str(tmp_path / "bias")
    common = f"model.image_size={IMAGE_SIZE},model.device=cpu,model.svd_components=8"
    monkeypatch.setattr(sys, "argv", [
        "analyze_positional_bias.py", "--config", os.path.join(ROOT, "configs", "base.yaml"),
        "--image-dir", str(imgs), "--calibration", cpath, "--pairs", "3", "--out", out,
        "--variant", f"hard:{common}", "--variant", f"none:{common},debias.mode=none",
        "--variant", f"random:{common},debias.mode=random",
        "--variant", f"pca:{common},debias.mode=pca,debias.calibration_path={cpath}"])
    # base config is loaded without the overrides: force a CPU / toy-size base too
    monkeypatch.setenv("INSID3_CALIBRATION_PATH", cpath)
    import insid3_ext.config as config
    orig_load = config.load_config

    def load(path, overrides=None):
        return orig_load(path, (overrides or []) + common.split(","))
    monkeypatch.setattr(config, "load_config", load)
    runpy.run_path(os.path.join(SCRIPTS, "analyze_positional_bias.py"), run_name="__main__")
    res = json.load(open(out + ".json"))
    v = res["variants"]
    assert set(v) == {"hard", "none", "random", "pca"} and res["pairs"] == 3
    assert abs(v["none"]["energy_kept"][0] - 1.0) < 1e-4 and v["hard"]["energy_kept"][0] < 1.0


def test_center_option(encoder, episode, calib, tmp_path):
    cal, _ = calib
    path = str(tmp_path / "c.pt")
    torch.save(cal, path)
    ext = INSID3Ext(copy.deepcopy(encoder), debias_cfg=DebiasConfig(mode="hard", center=True,
                                                                    calibration_path=path),
                    model_size="toy", **KW)
    ref, m, tgt = episode
    f = F.normalize(ext._extract_features(torch.cat([ref, tgt]).unsqueeze(0)), dim=2)
    mu = cal["feat_mean"].view(1, 1, -1, 1, 1)
    expected = INSID3._debias_features(ext, F.normalize(f - mu, dim=2))
    assert torch.allclose(ext._debias_features(f), expected, atol=1e-6)
    assert ext.predict_mask(ref, m, tgt).shape == (IMAGE_SIZE, IMAGE_SIZE)
    with pytest.raises(ConfigError):
        load_config(os.path.join(ROOT, "configs", "baseline.yaml"),
                    ["debias.center=true", "debias.calibration_path=null"])


def test_select_best_ignores_kshot_runs(tmp_path, monkeypatch):
    import yaml
    def run(exp, shots, miou):
        d = tmp_path / exp / "coco-tune" / f"fold0_shot{shots}_seed0"
        d.mkdir(parents=True)
        cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml")).to_dict()
        (d / "config.yaml").write_text(yaml.safe_dump(cfg))
        (d / "metrics.json").write_text(json.dumps({"exp_name": exp, "miou": miou, "shots": shots,
                                                    "complete": True, "n_episodes": 10}))
    run("baseline", 1, 50.0)
    run("t2_a", 1, 51.0)
    run("t2_b_5s", 5, 64.0)          # 5-shot score must not win the 1-shot selection
    run("t1_a", 1, 49.0)
    out = tmp_path / "sel"
    monkeypatch.syspath_prepend(SCRIPTS)
    monkeypatch.setattr(sys, "argv", ["select_best.py", "--root", str(tmp_path), "--out-dir", str(out)])
    runpy.run_path(os.path.join(SCRIPTS, "select_best.py"), run_name="__main__")
    assert "t2_a" in (out / "task2.yaml").read_text().splitlines()[0]


def test_check_consistency_cli(tmp_path):
    import subprocess
    def write(d, ious):
        d.mkdir(parents=True)
        with open(d / "episodes.jsonl", "w") as fp:
            for i, v in enumerate(ious):
                fp.write(json.dumps({"idx": i, "fingerprint": f"f{i}", "class_id": i % 3, "fg_iou": v,
                                     "inter": [0, v * 100], "union": [0, 100]}) + "\n")
    ious = list(np.linspace(0.1, 0.9, 100))
    write(tmp_path / "a", ious)
    write(tmp_path / "b", ious)
    write(tmp_path / "c", [v + (0.2 if i < 10 else 0) for i, v in enumerate(ious)])
    script = os.path.join(SCRIPTS, "check_consistency.py")
    ok = subprocess.run([sys.executable, script, "--ref", str(tmp_path / "a"), "--new", str(tmp_path / "b")])
    bad = subprocess.run([sys.executable, script, "--ref", str(tmp_path / "a"), "--new", str(tmp_path / "c")])
    assert ok.returncode == 0 and bad.returncode == 2


# ───────────── finding: the official basis is computed in training mode ─────────────

class TrainModeRandomEncoder(FakeEncoder):
    """Mimics DINOv3: in TRAINING mode the positional term is rescaled by a random factor in [1/2, 2]
    (like RoPE's rescale_coords); in eval mode it is deterministic."""

    def get_intermediate_layers(self, x, n=1, reshape=True):
        f = torch.tanh(self.proj(x))
        scale = float(torch.empty(1).uniform_(-0.693, 0.693).exp()) if self.training else 1.0
        return [f + self.pos * scale]


def _basis(cls, seed, **kw):
    torch.manual_seed(seed)                      # what the inference scripts do before building
    enc = TrainModeRandomEncoder()               # fresh hub model: training mode, like torch.hub.load
    return cls(enc, **KW, **kw)


def test_default_basis_reproduces_official_including_random_draw():
    for seed in (0, 1):
        assert torch.equal(_basis(INSID3, seed).positional_basis, _basis(INSID3Ext, seed).positional_basis)


def test_official_basis_depends_on_seed():
    from insid3_ext.debiasing import subspace_agreement
    a, b = _basis(INSID3, 0).positional_basis, _basis(INSID3, 1).positional_basis
    assert not torch.equal(a, b)
    assert subspace_agreement(a, b, 8) < 1 - 1e-6


def test_basis_eval_mode_is_deterministic_and_restores_mode():
    a = _basis(INSID3Ext, 0, basis_eval_mode=True)
    b = _basis(INSID3Ext, 1, basis_eval_mode=True)
    assert torch.equal(a.positional_basis, b.positional_basis)
    assert a.encoder.training                     # mode restored (the inference script calls .eval() later)
    torch.manual_seed(5)
    enc = TrainModeRandomEncoder().eval()
    ref = estimate_positional_svd(enc, IMAGE_SIZE, "cpu").U[:, :8]
    assert torch.equal(a.positional_basis, ref)
