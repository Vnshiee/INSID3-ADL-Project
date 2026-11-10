"""Model construction with configurable paths.

The official ``models/__init__.py`` loads DINOv3 with
``torch.hub.load("facebookresearch/dinov3", name, weights="pretrain/<file>")``, i.e.
it needs GitHub access at run time and a ``pretrain/`` folder in the CWD. Here:

* weights: ``model.weights_path`` or ``$INSID3_PRETRAIN_DIR/<official file name>``
  (default ``./pretrain``) — file names identical to the official ones;
* hub code: ``model.dinov3_repo`` or ``$INSID3_DINOV3_REPO`` → a local clone used with
  ``source='local'`` (offline servers); otherwise the GitHub hub entry point.

The INSID3 hyper-parameters are passed exactly like ``build_insid3_from_args``
(including ``resize_to_orig_size=False`` for evaluation).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .config import ExperimentConfig

HUB_NAMES = {"small": "dinov3_vits16", "base": "dinov3_vitb16", "large": "dinov3_vitl16"}
WEIGHT_FILES = {
    "small": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def resolve_weights_path(model_size: str, weights_path: Optional[str] = None) -> str:
    path = weights_path or os.path.join(os.environ.get("INSID3_PRETRAIN_DIR", "pretrain"),
                                        WEIGHT_FILES[model_size])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DINOv3 weights not found at '{path}'. Obtain them from "
            f"https://github.com/facebookresearch/dinov3 (licence form), then set "
            f"INSID3_PRETRAIN_DIR or model.weights_path. See docs/SERVER_SETUP.md.")
    return path


def build_encoder(model_size: str = "large", weights_path: Optional[str] = None,
                  dinov3_repo: Optional[str] = None) -> torch.nn.Module:
    """Load a DINOv3 ViT backbone through torch.hub (local clone if available)."""
    weights = resolve_weights_path(model_size, weights_path)
    repo = dinov3_repo or os.environ.get("INSID3_DINOV3_REPO")
    if repo:
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"INSID3_DINOV3_REPO='{repo}' is not a directory.")
        return torch.hub.load(repo, HUB_NAMES[model_size], source="local", weights=weights)
    return torch.hub.load("facebookresearch/dinov3", HUB_NAMES[model_size], weights=weights)


def build_model(cfg: ExperimentConfig, encoder: Optional[torch.nn.Module] = None):
    """Build a frozen :class:`INSID3Ext` from a config (encoder injectable for tests)."""
    from .model import INSID3Ext
    m = cfg.model
    if encoder is None:
        encoder = build_encoder(m.model_size, m.weights_path, m.dinov3_repo)
    model = INSID3Ext(
        encoder,
        debias_cfg=cfg.debias,
        proto_cfg=cfg.prototypes,
        model_size=m.model_size,
        full_diagnostics=cfg.run.full_diagnostics,
        basis_eval_mode=cfg.model.basis_eval_mode,
        image_size=m.image_size,
        svd_components=int(m.svd_components),
        tau=m.tau,
        merge_threshold=m.merge_threshold,
        mask_refiner=m.mask_refiner,
        crf_size=m.crf_size,
        resize_to_orig_size=False,
        device=m.device,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model
