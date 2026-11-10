"""Typed experiment configuration.

Design
------
* A config is a YAML file. ``_base_: other.yaml`` inherits (deep-merge, child wins);
  paths in ``_base_`` are relative to the including file.
* Strings may contain ``${env:VAR}`` or ``${env:VAR,default}``; they are expanded
  when the file is loaded, so no machine-specific path is ever hard-coded.
* Command-line overrides use dotted keys: ``--set debias.mode=weighted debias.alpha=0.5``.
  Values are parsed with ``yaml.safe_load`` (so ``0.5`` -> float, ``true`` -> bool).
* The merged dict is converted to dataclasses and validated by :func:`validate`.
  Invalid combinations raise :class:`ConfigError` *before* any GPU work starts.

Defaults reproduce the official INSID3 settings (paper Tab. 6 / ``opts.py``):
ViT-L, 1024 px, s = 500, tau = 0.6, alpha (merge threshold) = 0.2, hard projection,
single prototype.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


# ───────────────────────────── dataclasses ─────────────────────────────

@dataclass
class ModelConfig:
    model_size: str = "large"          # small | base | large  (DINOv3 ViT-S/B/L, patch 16)
    image_size: int = 1024             # images are resized to image_size x image_size
    svd_components: int = 500          # s in Eq. (4): rank of the hard positional projection
    tau: float = 0.6                   # agglomerative clustering similarity threshold
    merge_threshold: float = 0.2       # alpha in Eq. (14)
    mask_refiner: str = "bilinear"     # bilinear | crf  (paper Tab. 1 uses crf)
    crf_size: int = 640
    device: str = "cuda"
    weights_path: Optional[str] = None  # explicit .pth path; else ${INSID3_PRETRAIN_DIR}/<official name>
    dinov3_repo: Optional[str] = None   # local clone of facebookresearch/dinov3; else torch.hub github


@dataclass
class DebiasConfig:
    """Task 1. See docs/MATHEMATICAL_FORMULATION.md §2 for every formula."""
    mode: str = "hard"   # none | hard | weighted | spectral | calibrated | image | token
    rank: Optional[int] = None          # basis size r; None -> mode default (see debiasing.py)
    alpha: float = 1.0                  # weighted: lambda_k = alpha
    knee_rank: int = 500                # spectral: lambda = 0.5 at this (1-based) component
    sharpness: float = 1.0              # spectral: gamma; gamma -> inf recovers hard rank-knee projection
    strength: float = 1.0               # beta: multiplies the calibrated / Wiener gain (1 = MMSE point)
    calibration_path: Optional[str] = None
    variance_source: str = "calibration"  # image mode: calibration | noise
    template_source: str = "calibration"  # token mode: calibration | noise
    token_gain: str = "least_squares"     # token mode: fixed | least_squares
    max_gain: float = 2.0                 # token mode: clamp for least-squares gain


@dataclass
class PrototypeConfig:
    """Task 2. See docs/MATHEMATICAL_FORMULATION.md §3."""
    enabled: bool = False
    num_prototypes: int = 4             # M (k-means) or max M (agglomerative)
    clustering: str = "kmeans"          # kmeans | agglomerative
    cluster_space: str = "original"     # feature space used to *group* reference tokens
    ref_tau: float = 0.6                # agglomerative reference clustering threshold
    kmeans_iters: int = 50
    kmeans_restarts: int = 1
    min_points_per_prototype: int = 4   # M_eff = min(M, floor(N / this))
    min_cluster_fraction: float = 0.0   # drop clusters smaller than this fraction of N
    include_global: bool = False        # also keep the original mean prototype
    global_weight: float = 0.5          # its mass if included
    token_aggregation: str = "max"      # single | max | weighted | lse
    lse_temperature: float = 0.05
    region_matching: str = "prototype"  # prototype | chamfer | ot
    ot_epsilon: float = 0.05
    ot_rho: float = 0.5                 # KL relaxation of the prototype marginal (inf = balanced)
    ot_iters: int = 50
    seed: int = 0                       # local generator seed for k-means++ (never global RNG)


@dataclass
class DataConfig:
    dataset: str = "coco"
    data_root: str = "data"
    fold: int = 0
    shots: int = 1
    split: str = "test"                 # test (official protocol) | tune (COCO train2014, for model selection)
    tune_episodes: int = 1000
    num_workers: int = 0                # must stay 0: episodes are drawn from the global numpy RNG


@dataclass
class RunConfig:
    output_root: str = "outputs"
    exp_name: str = "baseline"
    seed: int = 0
    max_episodes: Optional[int] = None  # debug: stop after this many episodes
    resume: bool = True
    save_predictions: int = 0           # save the first N predicted masks (+ visualizations)
    log_every: int = 50
    profile_stages: bool = False        # per-stage timings (adds cuda syncs)


@dataclass
class CorrespondenceConfig:
    debiased: bool = True               # False = original DINOv3 features
    split: str = "test"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    debias: DebiasConfig = field(default_factory=DebiasConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ───────────────────────────── constants ─────────────────────────────

MODEL_SIZES = ("small", "base", "large")
DEBIAS_MODES = ("none", "hard", "weighted", "spectral", "calibrated", "image", "token")
SEG_DATASETS = ("coco", "lvis", "pascal_part", "paco_part", "isaid", "isic", "lung", "suim", "permis")
# Number of folds per dataset in the official protocol (README / opts.py).
DATASET_FOLDS = {"coco": 4, "lvis": 10, "pascal_part": 4, "paco_part": 4, "isaid": 3,
                 "isic": 1, "lung": 1, "suim": 1, "permis": 1}
TOKEN_AGGREGATIONS = ("single", "max", "weighted", "lse")
REGION_MATCHINGS = ("prototype", "chamfer", "ot")


# ───────────────────────────── loading ─────────────────────────────

_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${env:VAR,default}`` in strings."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            if var in os.environ:
                return os.environ[var]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable ${var} is not set and has no default "
                              f"(see env.example.sh).")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` (inputs are not mutated)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_bases(path: str, _seen: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load a YAML file, resolving ``_base_`` inheritance (single or list)."""
    path = os.path.abspath(path)
    _seen = list(_seen or [])
    if path in _seen:
        raise ConfigError(f"Circular _base_ include: {' -> '.join(_seen + [path])}")
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping.")
    bases = data.pop("_base_", None)
    merged: Dict[str, Any] = {}
    if bases:
        for b in [bases] if isinstance(bases, str) else bases:
            bpath = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
            merged = deep_merge(merged, load_yaml_with_bases(bpath, _seen + [path]))
    return deep_merge(merged, data)


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides. Unknown keys are caught later by :func:`from_dict`."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot set {key}: {p} is not a section.")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    """Convert a nested dict into :class:`ExperimentConfig`, rejecting unknown keys."""
    sections = {f.name: f.type for f in dataclasses.fields(ExperimentConfig)}
    classes = {"model": ModelConfig, "debias": DebiasConfig, "prototypes": PrototypeConfig,
               "data": DataConfig, "run": RunConfig, "correspondence": CorrespondenceConfig}
    unknown = set(d) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
    kwargs = {}
    for name, cls in classes.items():
        sec = d.get(name, {}) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        bad = set(sec) - valid
        if bad:
            raise ConfigError(f"Unknown keys in [{name}]: {sorted(bad)}. Valid: {sorted(valid)}")
        kwargs[name] = cls(**sec)
    return ExperimentConfig(**kwargs)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """Load, merge, expand env vars, apply overrides, convert and validate."""
    raw = load_yaml_with_bases(path) if path else {}
    raw = apply_overrides(raw, overrides)
    raw = _expand_env(raw)
    cfg = from_dict(raw)
    validate(cfg)
    return cfg


def save_config(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w") as fp:
        yaml.safe_dump(cfg.to_dict(), fp, sort_keys=False)


# ───────────────────────────── validation ─────────────────────────────

def needs_calibration(d: DebiasConfig) -> bool:
    """True if the debias mode reads statistics from a calibration file."""
    return (d.mode == "calibrated"
            or (d.mode == "image" and d.variance_source == "calibration")
            or (d.mode == "token" and d.template_source == "calibration"))


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(cfg: ExperimentConfig) -> None:
    """Check ranges and cross-field consistency. Raises :class:`ConfigError`."""
    m, d, p, data, run = cfg.model, cfg.debias, cfg.prototypes, cfg.data, cfg.run

    _check(m.model_size in MODEL_SIZES, f"model.model_size must be one of {MODEL_SIZES}")
    _check(m.image_size > 0 and m.image_size % 16 == 0,
           "model.image_size must be a positive multiple of the DINOv3 patch size (16)")
    _check(m.svd_components >= 0, "model.svd_components must be >= 0")
    _check(0.0 < m.tau < 1.0, "model.tau must lie in (0, 1)")
    _check(m.mask_refiner in ("bilinear", "crf"), "model.mask_refiner must be bilinear | crf")

    _check(d.mode in DEBIAS_MODES, f"debias.mode must be one of {DEBIAS_MODES}")
    _check(d.rank is None or d.rank >= 1, "debias.rank must be >= 1 or null")
    _check(0.0 <= d.alpha <= 1.0, "debias.alpha must lie in [0, 1] (1 = hard projection)")
    _check(d.knee_rank >= 1, "debias.knee_rank must be >= 1")
    _check(d.sharpness > 0, "debias.sharpness must be > 0")
    _check(d.strength >= 0, "debias.strength must be >= 0")
    _check(d.variance_source in ("calibration", "noise"), "debias.variance_source: calibration | noise")
    _check(d.template_source in ("calibration", "noise"), "debias.template_source: calibration | noise")
    _check(d.token_gain in ("fixed", "least_squares"), "debias.token_gain: fixed | least_squares")
    _check(d.max_gain > 0, "debias.max_gain must be > 0")
    _check(not needs_calibration(d) or bool(d.calibration_path),
           f"debias.mode={d.mode} needs debias.calibration_path "
           f"(create it with scripts/calibrate_positional.py)")

    _check(p.num_prototypes >= 1, "prototypes.num_prototypes must be >= 1")
    _check(p.clustering in ("kmeans", "agglomerative"), "prototypes.clustering: kmeans | agglomerative")
    _check(p.cluster_space in ("original", "debiased"), "prototypes.cluster_space: original | debiased")
    _check(0.0 < p.ref_tau < 1.0, "prototypes.ref_tau must lie in (0, 1)")
    _check(p.kmeans_iters >= 1 and p.kmeans_restarts >= 1, "k-means iterations/restarts must be >= 1")
    _check(p.min_points_per_prototype >= 1, "prototypes.min_points_per_prototype must be >= 1")
    _check(0.0 <= p.min_cluster_fraction < 1.0, "prototypes.min_cluster_fraction must lie in [0, 1)")
    _check(0.0 < p.global_weight < 1.0, "prototypes.global_weight must lie in (0, 1)")
    _check(p.token_aggregation in TOKEN_AGGREGATIONS, f"prototypes.token_aggregation: {TOKEN_AGGREGATIONS}")
    _check(p.lse_temperature > 0, "prototypes.lse_temperature must be > 0")
    _check(p.region_matching in REGION_MATCHINGS, f"prototypes.region_matching: {REGION_MATCHINGS}")
    _check(p.ot_epsilon > 0 and p.ot_rho > 0 and p.ot_iters >= 1, "OT epsilon/rho must be > 0, iters >= 1")

    _check(data.dataset in SEG_DATASETS + ("spair",), f"data.dataset must be one of {SEG_DATASETS + ('spair',)}")
    _check(data.shots >= 1, "data.shots must be >= 1")
    _check(data.split in ("test", "tune"), "data.split: test | tune")
    _check(data.split != "tune" or data.dataset == "coco",
           "data.split=tune is only implemented for COCO (train2014 images, disjoint from COCO-20i test)")
    if data.dataset in DATASET_FOLDS and data.split == "test":
        _check(0 <= data.fold < DATASET_FOLDS[data.dataset],
               f"data.fold for {data.dataset} must be in [0, {DATASET_FOLDS[data.dataset] - 1}]")
    _check(data.num_workers == 0,
           "data.num_workers must be 0: the official datasets sample episodes from the global numpy RNG, "
           "and worker processes would change the episode sequence")

    _check(run.save_predictions >= 0, "run.save_predictions must be >= 0")
    _check(run.max_episodes is None or run.max_episodes >= 1, "run.max_episodes must be >= 1 or null")
