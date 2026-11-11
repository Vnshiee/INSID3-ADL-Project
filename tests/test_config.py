"""Configuration loading, inheritance, overrides, env expansion and validation."""

import glob
import os

import pytest

from conftest import ROOT
from insid3_ext.config import ConfigError, load_config

CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True))


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.relpath(p, ROOT) for p in CONFIGS])
def test_every_shipped_config_validates(path):
    cfg = load_config(path)
    assert cfg.run.exp_name


def test_defaults_are_official():
    cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"))
    assert (cfg.model.model_size, cfg.model.image_size, cfg.model.svd_components) == ("large", 1024, 500)
    assert (cfg.model.tau, cfg.model.merge_threshold) == (0.6, 0.2)
    assert cfg.debias.mode == "hard" and not cfg.prototypes.enabled
    assert cfg.run.seed == 0


def test_overrides_and_types():
    cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"),
                      ["debias.mode=weighted", "debias.alpha=0.5", "prototypes.enabled=true", "data.fold=3"])
    assert cfg.debias.alpha == 0.5 and cfg.prototypes.enabled is True and cfg.data.fold == 3


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("INSID3_DATA_ROOT", "/some/data")
    cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"))
    assert cfg.data.data_root == "/some/data"


@pytest.mark.parametrize("bad", [
    ["debias.mode=soft"], ["debias.alpha=1.5"], ["model.image_size=1000"], ["data.fold=4"],
    ["prototypes.token_aggregation=mean"], ["prototypes.num_prototypes=0"],
    ["data.split=tune", "data.dataset=lvis"], ["debias.mode=calibrated", "debias.calibration_path=null"],
    ["model.tau=1.0"], ["prototypes.ot_rho=0"],
])
def test_invalid_values_rejected(bad):
    with pytest.raises(ConfigError):
        load_config(os.path.join(ROOT, "configs", "baseline.yaml"), bad)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(os.path.join(ROOT, "configs", "baseline.yaml"), ["debias.alhpa=0.5"])


def test_circular_base(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "a.yaml"))
