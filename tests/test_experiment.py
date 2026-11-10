"""Experiment framework: end-to-end CPU run on a synthetic dataset, resume, metrics."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import IMAGE_SIZE, ROOT, FakeEncoder, make_image, make_mask
from insid3_ext import datasets_ext, experiment
from insid3_ext.builder import build_model
from insid3_ext.config import load_config


class ToyDataset:
    """Mimics the official episode dict and draws episodes from the global numpy RNG."""
    class_ids = [0, 1]

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        c = int(np.random.choice(self.class_ids))
        off = int(np.random.randint(0, 40))
        tb = (off, off, off + 56, off + 56)
        unnorm = lambda x: x  # noqa: E731 (tensors are passed straight to the model)
        return {"ref_imgs": [unnorm(make_image(box=(16, 16, 64, 64), seed=idx))],
                "ref_masks": [make_mask((16, 16, 64, 64)).float()],
                "tgt_img": make_image(box=tb, seed=100 + idx), "tgt_mask": make_mask(tb).float(),
                "class_id": torch.tensor(c)}


def _cfg(tmp_path, *extra):
    return load_config(os.path.join(ROOT, "configs", "baseline.yaml"),
                       [f"run.output_root={tmp_path}", "model.device=cpu", f"model.image_size={IMAGE_SIZE}",
                        "model.svd_components=8", "run.log_every=1000", *extra])


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(datasets_ext, "build_eval_dataset", lambda cfg: ToyDataset())


def _model(cfg):
    return build_model(cfg, encoder=FakeEncoder().eval())


def test_end_to_end_and_resume(tmp_path, toy):
    cfg = _cfg(tmp_path, "run.save_predictions=1")
    m1 = experiment.run_segmentation(cfg, model=_model(cfg))
    out = experiment.run_dir(cfg)
    recs = experiment.read_jsonl(os.path.join(out, "episodes.jsonl"))
    assert len(recs) == 6 and m1["complete"] and 0 <= m1["miou"] <= 100
    assert os.path.isfile(os.path.join(out, "predictions", "00000.npz"))
    for f in ("config.yaml", "env.json", "metrics.json", "log.txt"):
        assert os.path.isfile(os.path.join(out, f))

    # resume: nothing recomputed, identical metrics
    m2 = experiment.run_segmentation(_cfg(tmp_path), model=_model(cfg))
    assert len(experiment.read_jsonl(os.path.join(out, "episodes.jsonl"))) == 6
    assert m2["miou"] == pytest.approx(m1["miou"])


def test_partial_resume_matches_full_run(tmp_path, toy):
    cfg_full = _cfg(tmp_path / "a")
    full = experiment.run_segmentation(cfg_full, model=_model(cfg_full))
    cfg_part = _cfg(tmp_path / "b", "run.max_episodes=3")
    experiment.run_segmentation(cfg_part, model=_model(cfg_part))
    cfg_rest = _cfg(tmp_path / "b")
    rest = experiment.run_segmentation(cfg_rest, model=_model(cfg_rest))
    assert rest["miou"] == pytest.approx(full["miou"])


def test_fingerprint_mismatch_detected(tmp_path, toy):
    cfg = _cfg(tmp_path, "run.max_episodes=2")
    experiment.run_segmentation(cfg, model=_model(cfg))
    bad = _cfg(tmp_path, "run.seed=1")             # different seed -> different episodes...
    bad.run.exp_name = cfg.run.exp_name
    path = os.path.join(experiment.run_dir(cfg), "episodes.jsonl")
    target = experiment.run_dir(bad)
    os.makedirs(target, exist_ok=True)
    with open(path) as src, open(os.path.join(target, "episodes.jsonl"), "w") as dst:
        dst.write(src.read())                      # ...but stale records copied in
    with pytest.raises(RuntimeError, match="fingerprint"):
        experiment.run_segmentation(bad, model=_model(bad))


def test_methods_see_identical_episodes(tmp_path, toy):
    a = _cfg(tmp_path, "run.exp_name=a")
    b = _cfg(tmp_path, "run.exp_name=b", "prototypes.enabled=true", "prototypes.num_prototypes=3")
    experiment.run_segmentation(a, model=_model(a))
    experiment.run_segmentation(b, model=_model(b))
    fa = [r["fingerprint"] for r in experiment.read_jsonl(os.path.join(experiment.run_dir(a), "episodes.jsonl"))]
    fb = [r["fingerprint"] for r in experiment.read_jsonl(os.path.join(experiment.run_dir(b), "episodes.jsonl"))]
    assert fa == fb
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compare_runs.py"), "--a",
                          experiment.run_dir(a), "--b", experiment.run_dir(a), "--bootstrap", "200"],
                         capture_output=True, text=True, check=True)
    assert json.loads(res.stdout)["delta"] == 0.0


def test_metrics_match_official_meter():
    from utils.metrics import AverageMeter, Evaluator
    g = torch.Generator().manual_seed(0)
    meter = AverageMeter("coco", [0, 1, 2], device="cpu")
    recs = []
    for i in range(12):
        p = torch.rand(16, 16, generator=g) > 0.5
        t = torch.rand(16, 16, generator=g) > 0.5
        inter, union = Evaluator.classify_prediction(p, t)
        c = i % 3
        meter.update(inter, union, torch.tensor([c]))
        recs.append({"inter": inter.flatten().tolist(), "union": union.flatten().tolist(), "class_id": c})
    m = experiment.compute_seg_metrics(recs, [0, 1, 2], "coco")
    assert m["miou"] == pytest.approx(float(meter.compute_iou()[0]))
    assert m["fbiou"] == pytest.approx(float(meter.compute_iou()[1]))


def test_pck_matches_official_evaluator():
    from utils.metrics import PCKEvaluator
    g = torch.Generator().manual_seed(0)
    ev = PCKEvaluator(pck_by="image")
    recs = []
    for i in range(5):
        trg = torch.rand(7, 2, generator=g) * 100
        pred = trg + torch.randn(7, 2, generator=g) * 10
        thr = 80.0
        ev.calculate_pck(trg[None], pred[None], torch.tensor([7]), ["cat"], torch.tensor([thr]))
        recs.append({"dist": torch.norm(trg - pred, dim=-1).tolist(), "pck_thresh": thr, "category": "cat"})
    ours = experiment.compute_pck(recs)
    ref = ev.get_result()
    assert [ours[k] for k in ("pck@0.05", "pck@0.1", "pck@0.15", "pck@0.2")] == \
        pytest.approx([100 * v for v in ref])


def test_read_jsonl_tolerates_truncated_tail(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"idx": 0}\n{"idx": 1}\n{"idx": 2, "inter"')
    assert [r["idx"] for r in experiment.read_jsonl(str(p))] == [0, 1]
