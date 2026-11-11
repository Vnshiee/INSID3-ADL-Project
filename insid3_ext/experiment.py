"""Resumable evaluation loops.

Protocol fidelity
-----------------
* Same seeding as the official scripts, then model construction, then the dataset.
* Episodes are fetched one after another with ``ds[idx]``, in this process. This is what the official
  ``DataLoader(batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])`` does, so the
  global-numpy-RNG episode sequence is identical to the official one. (Worker processes would each have
  their own copy of the random generator and change the episodes, so we never use them.)
* Prediction at ``image_size`` resolution, GT resized with nearest interpolation, IoU via
  the official ``Evaluator`` / ``AverageMeter`` (class-wise mIoU and FB-IoU).
* None of our modules touch the global RNGs, so every method sees *the same episodes*;
  each episode carries a fingerprint so this can be verified and paired statistics used.

Output layout:  <output_root>/<exp_name>/<dataset>[-tune]/fold<f>_shot<k>_seed<s>/
    config.yaml  env.json  log.txt  episodes.jsonl  metrics.json  predictions/
Resume: completed episodes in episodes.jsonl are skipped (their data are still drawn
from the dataset to keep the RNG stream aligned, and fingerprints are re-checked).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ExperimentConfig, save_config
from .runtime import (CudaTimer, environment_report, peak_memory_mb, reset_peak_memory,
                      resolve_device, set_seed)


# ───────────────────────────── helpers ─────────────────────────────

def run_dir(cfg: ExperimentConfig, task: str = "seg") -> str:
    d = cfg.data
    if task == "corr":
        sub = os.path.join("spair", f"{cfg.correspondence.split}_seed{cfg.run.seed}")
    else:
        name = d.dataset + ("-tune" if d.split == "tune" else "")
        sub = os.path.join(name, f"fold{d.fold}_shot{d.shots}_seed{cfg.run.seed}")
    return os.path.join(cfg.run.output_root, cfg.run.exp_name, sub)


def _tensor_digest(t: torch.Tensor) -> str:
    """A short hash of a binary mask: equal masks give equal hashes."""
    a = (t > 0).cpu().numpy().astype(np.uint8)             # the mask as 0/1
    data = np.packbits(a).tobytes() + str(a.shape).encode()  # its bits and its shape, as bytes
    return hashlib.sha1(data).hexdigest()[:12]              # first 12 hex characters are enough


def episode_fingerprint(batch: Dict[str, Any]) -> str:
    """Hash of class id, target mask and reference masks (identifies the episode)."""
    parts = [str(int(batch["class_id"])), _tensor_digest(torch.as_tensor(batch["tgt_mask"]))]
    parts += [_tensor_digest(torch.as_tensor(m)) for m in batch["ref_masks"]]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read the per-episode records.

    A run that was interrupted (Ctrl-C, crash) can leave one half-written line. That line is skipped and
    reading CONTINUES, so the episodes written after a resume are not lost. If an episode appears twice,
    the later record wins.
    """
    if not os.path.isfile(path):
        return []
    by_idx: Dict[Any, Dict[str, Any]] = {}
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                       # half-written line from an interrupted run
            by_idx[rec.get("idx", len(by_idx))] = rec
    return list(by_idx.values())


def _open_for_append(path: str):
    """Open the episode log for appending; first finish a half-written last line with a newline, so the
    next record starts on its own line instead of being glued to the broken one."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fp:
            fp.seek(-1, os.SEEK_END)
            ends_with_newline = fp.read(1) == b"\n"
        if not ends_with_newline:
            with open(path, "a") as fp:
                fp.write("\n")
    return open(path, "a")


def _log(path: str, msg: str) -> None:
    print(msg)
    with open(path, "a") as fp:
        fp.write(msg + "\n")


def _prepare_run(cfg: ExperimentConfig, task: str):
    cfg.model.device = resolve_device(cfg.model.device)
    set_seed(cfg.run.seed)
    out = run_dir(cfg, task)
    os.makedirs(out, exist_ok=True)
    jsonl = os.path.join(out, "episodes.jsonl")
    if not cfg.run.resume and os.path.exists(jsonl):
        os.replace(jsonl, jsonl + ".bak")
    save_config(cfg, os.path.join(out, "config.yaml"))
    with open(os.path.join(out, "env.json"), "w") as fp:
        json.dump(environment_report(), fp, indent=2)
    log = os.path.join(out, "log.txt")
    _log(log, " ".join(sys.argv))
    return out, jsonl, log


# ───────────────────────────── segmentation ─────────────────────────────

def compute_seg_metrics(records: List[Dict[str, Any]], class_ids, dataset: str) -> Dict[str, float]:
    """Official mIoU / FB-IoU from per-episode intersections and unions."""
    from utils.metrics import AverageMeter
    meter = AverageMeter(dataset, list(class_ids), device="cpu")
    for r in records:
        meter.update(torch.tensor(r["inter"]).view(2, 1), torch.tensor(r["union"]).view(2, 1),
                     torch.tensor([r["class_id"]]))
    miou, fbiou = meter.compute_iou()
    return {"miou": float(miou), "fbiou": float(fbiou)}


def run_segmentation(cfg: ExperimentConfig, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """Evaluate one (dataset, fold, shots, seed) configuration. Returns metrics."""
    from utils.metrics import Evaluator
    from .builder import build_model
    from .datasets_ext import build_eval_dataset, image_overlap

    out, jsonl, log = _prepare_run(cfg, "seg")
    device = cfg.model.device

    if model is None:
        model = build_model(cfg)
    model.to(device)
    model.eval()
    ds = build_eval_dataset(cfg)
    if cfg.data.split == "tune":
        _log(log, f"[tune] images shared with COCO-20i test folds: "
                  f"{image_overlap(ds, cfg.data.data_root, cfg.data.shots)} (expected 0)")

    done = {r["idx"]: r for r in read_jsonl(jsonl)}
    n_total = len(ds) if cfg.run.max_episodes is None else min(len(ds), cfg.run.max_episodes)
    _log(log, f"run_dir={out}  episodes={n_total}  already_done={len(done)}")
    pred_dir = os.path.join(out, "predictions")

    with _open_for_append(jsonl) as fp:
        for idx in tqdm(range(n_total), ncols=80):
            batch = ds[idx]                       # always drawn: keeps the RNG stream aligned
            fpr = episode_fingerprint(batch)
            if idx in done:
                if done[idx].get("fingerprint") != fpr:
                    raise RuntimeError(f"Episode {idx} fingerprint mismatch on resume "
                                       f"({done[idx].get('fingerprint')} vs {fpr}); the episode "
                                       f"sequence changed. Re-run with run.resume=false.")
                continue

            reset_peak_memory(device)
            for ref_img, ref_mask in zip(batch["ref_imgs"], batch["ref_masks"]):
                model.set_reference(ref_img, ref_mask)
            model.set_target(batch["tgt_img"])
            with CudaTimer(device) as t:
                pred_mask = model.segment()
            tgt_mask = F.interpolate(batch["tgt_mask"].unsqueeze(0).unsqueeze(0).float(),
                                     size=pred_mask.shape, mode="nearest").squeeze(0).squeeze(0) > 0.5
            inter, union = Evaluator.classify_prediction(pred_mask, tgt_mask,
                                                         tgt_ignore_idx=batch.get("tgt_ignore_idx"))
            rec = {"idx": idx, "fingerprint": fpr, "class_id": int(batch["class_id"]),
                   "inter": inter.flatten().cpu().tolist(), "union": union.flatten().cpu().tolist(),
                   "fg_iou": float(inter[1] / max(float(union[1]), 1.0)),
                   "time_ms": t.ms, "peak_mem_mb": peak_memory_mb(device),
                   "pred_area": int(pred_mask.sum()), "gt_area": int(tgt_mask.sum())}
            # Extra numbers logged by our model (INSID3Ext); the plain official model has none.
            if hasattr(model, "last_diagnostics"):
                rec.update(model.last_diagnostics)                   # candidates, seed, scores, ...
            if hasattr(model, "debiaser") and model.debiaser.last_stats:
                rec["debias_stats"] = model.debiaser.last_stats     # mean lambda / gamma of this episode
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            done[idx] = rec

            if idx < cfg.run.save_predictions:
                _save_prediction(pred_dir, idx, batch, pred_mask)
            if (idx + 1) % cfg.run.log_every == 0:
                m = compute_seg_metrics(list(done.values()), ds.class_ids, cfg.data.dataset)
                tqdm.write(f"[{idx + 1}/{n_total}] mIoU={m['miou']:.2f}")

    records = [done[i] for i in sorted(done) if i < n_total]
    metrics = compute_seg_metrics(records, ds.class_ids, cfg.data.dataset)
    times = [r["time_ms"] for r in records]
    # peak_mem_mb is NaN on the CPU (no GPU memory to measure); leave those out
    mems = [r["peak_mem_mb"] for r in records if not math.isnan(r["peak_mem_mb"])]
    metrics.update({
        "task": "segmentation", "exp_name": cfg.run.exp_name, "dataset": cfg.data.dataset,
        "split": cfg.data.split, "fold": cfg.data.fold, "shots": cfg.data.shots, "seed": cfg.run.seed,
        "debias_mode": cfg.debias.mode, "prototypes": cfg.prototypes.enabled,
        "n_episodes": len(records), "n_expected": len(ds), "complete": len(records) == len(ds),
        "time_ms_mean": statistics.fmean(times) if times else None,
        "time_ms_median": statistics.median(times) if times else None,
        "peak_mem_mb_max": max(mems) if mems else None,
    })
    with open(os.path.join(out, "metrics.json"), "w") as fpm:
        json.dump(metrics, fpm, indent=2)
    _log(log, f"mIoU = {metrics['miou']:.2f}  FB-IoU = {metrics['fbiou']:.2f}  "
              f"({len(records)}/{len(ds)} episodes)")
    return metrics


def _save_prediction(pred_dir: str, idx: int, batch: Dict[str, Any], pred_mask: torch.Tensor) -> None:
    os.makedirs(pred_dir, exist_ok=True)
    m = pred_mask.cpu().numpy().astype(bool)
    np.savez_compressed(os.path.join(pred_dir, f"{idx:05d}.npz"), mask=np.packbits(m), shape=m.shape)
    try:
        from utils.visualization import visualize_prediction_segmentation
        visualize_prediction_segmentation(batch["ref_imgs"][0], batch["ref_masks"][0], batch["tgt_img"],
                                          pred_mask.cpu(), os.path.join(pred_dir, f"{idx:05d}.png"))
    except Exception as e:  # visualization must never kill an evaluation run
        print(f"[warn] visualization of episode {idx} failed: {e}")


# ───────────────────────────── semantic correspondence ─────────────────────────────

PCK_ALPHAS = (0.05, 0.1, 0.15, 0.2)


def compute_pck(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Official image-wise PCK (utils.metrics.PCKEvaluator) from stored distances."""
    from utils.metrics import PCKEvaluator
    ev = PCKEvaluator(pck_by="image")
    for r in records:
        d = torch.tensor(r["dist"]).view(1, -1, 1)
        zeros = torch.zeros_like(d)
        # calculate_pck uses ||trg - pred||; feed (dist, 0) pairs to reproduce it exactly.
        ev.calculate_pck(d, zeros, torch.tensor([len(r["dist"])]), [r["category"]],
                         torch.tensor([r["pck_thresh"]]))
    res = ev.get_result()
    return {f"pck@{a}": 100.0 * float(v) for a, v in zip(PCK_ALPHAS, res)}


def run_correspondence(cfg: ExperimentConfig, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """SPair-71k keypoint transfer (official protocol of inference_keypoint_matching.py)."""
    from datasets import build_dataset
    from .builder import build_model
    from .datasets_ext import make_args

    out, jsonl, log = _prepare_run(cfg, "corr")
    device = cfg.model.device
    if model is None:
        model = build_model(cfg)
    model.to(device)
    model.eval()
    ds = build_dataset("spair", args=make_args(cfg))
    done = {r["idx"]: r for r in read_jsonl(jsonl)}
    n_total = len(ds) if cfg.run.max_episodes is None else min(len(ds), cfg.run.max_episodes)
    _log(log, f"run_dir={out}  pairs={n_total}  already_done={len(done)}  debiased={cfg.correspondence.debiased}")

    with _open_for_append(jsonl) as fp:
        for idx in tqdm(range(n_total), ncols=80):
            if idx in done:
                continue
            b = ds[idx]
            model.set_reference(b["src_img"])
            model.set_target(b["trg_img"])
            with CudaTimer(device) as t:
                pred = model.match(b["src_kps"].to(device), use_debiased=cfg.correspondence.debiased)
            n = int(b["n_pts"])
            dist = torch.norm(b["trg_kps"][:n].to(pred.device) - pred[:n], dim=-1)
            rec = {"idx": idx, "category": b["category"], "pck_thresh": float(b["pck_thresh"]),
                   "dist": dist.cpu().tolist(), "time_ms": t.ms}
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            done[idx] = rec

    records = [done[i] for i in sorted(done) if i < n_total]
    metrics = compute_pck(records)
    metrics.update({"task": "correspondence", "exp_name": cfg.run.exp_name, "split": cfg.correspondence.split,
                    "debiased": cfg.correspondence.debiased, "debias_mode": cfg.debias.mode,
                    "model_size": cfg.model.model_size, "image_size": cfg.model.image_size,
                    "svd_components": cfg.model.svd_components, "n_pairs": len(records),
                    "n_expected": len(ds), "complete": len(records) == len(ds)})
    with open(os.path.join(out, "metrics.json"), "w") as fpm:
        json.dump(metrics, fpm, indent=2)
    _log(log, "  ".join(f"{k}={v:.2f}" for k, v in metrics.items() if k.startswith("pck")))
    return metrics
