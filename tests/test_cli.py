"""Execute the command-line scripts end-to-end with the fake encoder and a toy dataset.

Only ``build_encoder`` (no DINOv3 weights on CPU) and the dataset builder are patched;
argument parsing, config loading, the experiment loop, aggregation, selection and
comparison all run for real.
"""

import glob
import json
import os
import runpy
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import IMAGE_SIZE, ROOT, FakeEncoder
from test_experiment import ToyDataset

SCRIPTS = os.path.join(ROOT, "scripts")
COMMON = [f"model.image_size={IMAGE_SIZE}", "model.svd_components=8", "model.device=cpu", "run.log_every=1000"]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    import insid3_ext.builder as builder
    import insid3_ext.datasets_ext as dext
    monkeypatch.setattr(builder, "build_encoder", lambda *a, **k: FakeEncoder().eval())
    monkeypatch.setattr(dext, "build_eval_dataset", lambda cfg: ToyDataset())
    monkeypatch.setattr(dext, "image_overlap", lambda *a, **k: 0)
    monkeypatch.syspath_prepend(SCRIPTS)
    return tmp_path


def _run(script, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script] + args)
    runpy.run_path(os.path.join(SCRIPTS, script), run_name="__main__")


def test_segmentation_cli_select_aggregate_compare(patched, monkeypatch):
    out = str(patched / "out")
    runs = [("configs/baseline.yaml", "baseline"), ("configs/task1/weighted.yaml", "t1_weighted_a0.5"),
            ("configs/task2/max.yaml", "t2_max_m3")]
    for cfg, name in runs:
        extra = ["debias.alpha=0.5"] if "weighted" in cfg else (["prototypes.num_prototypes=3"] if "max" in cfg else [])
        for split in ("tune", "test"):
            _run("run_segmentation.py", ["--config", os.path.join(ROOT, cfg), "--set", *COMMON, *extra,
                                         f"run.output_root={out}", f"run.exp_name={name}", f"data.split={split}",
                                         "data.dataset=coco"], monkeypatch)
    assert len(glob.glob(os.path.join(out, "*", "*", "*", "metrics.json"))) == 6

    sel = str(patched / "selected")
    _run("select_best.py", ["--root", out, "--out-dir", sel], monkeypatch)
    assert {"task1.yaml", "task2.yaml", "combined.yaml", "selection_report.md"} <= set(os.listdir(sel))
    from insid3_ext.config import load_config
    comb = load_config(os.path.join(sel, "combined.yaml"))
    assert comb.debias.mode == "weighted" and comb.prototypes.enabled and comb.prototypes.num_prototypes == 3

    res = str(patched / "results" / "summary")
    _run("aggregate_results.py", ["--root", out, "--out", res], monkeypatch)
    text = open(res + ".md").read()
    assert "baseline" in text and "t2_max_m3" in text

    cmp = str(patched / "results" / "cmp")
    _run("compare_all.py", ["--root", out, "--out", cmp, "--bootstrap", "100"], monkeypatch)
    rows = json.load(open(cmp + ".json"))
    assert {r["exp"] for r in rows} == {"t1_weighted_a0.5", "t2_max_m3"}

    fail = str(patched / "results" / "fail")
    _run("analyze_failures.py", ["--a", os.path.join(out, "baseline", "coco", "fold0_shot1_seed0"),
                                 "--b", os.path.join(out, "t2_max_m3", "coco", "fold0_shot1_seed0"),
                                 "--out", fail], monkeypatch)
    assert os.path.isfile(fail + ".csv")


def test_calibration_cli(patched, monkeypatch):
    imgs = patched / "imgs"
    imgs.mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        Image.fromarray(rng.integers(0, 255, (100, 140, 3), dtype=np.uint8)).save(imgs / f"{i}.jpg")
    out = str(patched / "calib.pt")
    _run("calibrate_positional.py", ["--config", os.path.join(ROOT, "configs", "base.yaml"), "--set", *COMMON,
                                     "--image-dir", str(imgs), "--num-images", "6", "--batch-size", "4",
                                     "--out", out], monkeypatch)
    import torch
    cal = torch.load(out, weights_only=False)
    assert cal["meta"]["num_images"] == 6 and cal["meta"]["image_size"] == IMAGE_SIZE
    assert os.path.isfile(str(patched / "calib_summary.json"))


def test_verify_upstream_script():
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "verify_upstream.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_shell_scripts_parse():
    for f in glob.glob(os.path.join(ROOT, "*.sh")):
        assert subprocess.run(["bash", "-n", f]).returncode == 0, f



def test_model_figures(patched, monkeypatch):
    """visualize.py similarity / prototypes run end to end (fake encoder, CPU)."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    for name in ("r.jpg", "t.jpg"):
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(patched / name)
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[20:70, 30:100] = 255
    Image.fromarray(mask).save(patched / "m.png")
    cfg = patched / "c.yaml"           # Task-2 config at the fake encoder's size
    cfg.write_text("_base_: " + os.path.join(ROOT, "configs", "task2", "max.yaml") + "\n"
                   "model: {device: cpu, image_size: 128, svd_components: 8}\n")
    images = ["--ref-image", str(patched / "r.jpg"), "--ref-mask", str(patched / "m.png"),
              "--target", str(patched / "t.jpg")]
    _run("visualize.py", ["similarity", "--configs", str(cfg), *images, "--out", str(patched / "sim.png")],
         monkeypatch)
    _run("visualize.py", ["prototypes", "--config", str(cfg), *images, "--out", str(patched / "protos.png")],
         monkeypatch)
    assert (patched / "sim.png").exists() and (patched / "protos.png").exists()
