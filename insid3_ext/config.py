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

# Every default below either (1) copies the official INSID3 value (paper Tab. 6 / opts.py), or
# (2) is a STARTING value for one of our new options that the tuning sweeps (run_task1.sh,
# run_task2.sh, run_revision.sh) vary; the value finally used is chosen on the COCO tuning split only.
# The comment on each line says which of the two it is and why.

@dataclass
class ModelConfig:
    model_size: str = "large"           # official: DINOv3 ViT-L/16 (paper Sec. 4). small | base | large
    image_size: int = 1024              # official: images resized to 1024x1024 -> 64x64 patches
    svd_components: int = 500           # official s (paper Tab. 6): directions removed by the hard projection
    basis_eval_mode: bool = False       # official = False: the basis is computed while DINOv3 is still in
                                        # training mode, where its RoPE rescales patch coordinates by a random
                                        # factor in [1/2, 2] (see model.py). True = compute it in eval mode
                                        # (no rescaling, like every real feature). Experiment t1_hard_evalbasis.
    tau: float = 0.6                    # official: clustering threshold (part-level granularity, paper Fig. 7)
    merge_threshold: float = 0.2        # official alpha (paper Tab. 6): final cluster score must exceed it
    mask_refiner: str = "bilinear"      # official script default; paper Tab. 1 numbers use "crf"
    crf_size: int = 640                 # official CRF resolution
    device: str = "cuda"
    weights_path: Optional[str] = None  # where the DINOv3 .pth is (default: $INSID3_PRETRAIN_DIR/<official name>)
    dinov3_repo: Optional[str] = None   # local clone of the DINOv3 code (works without internet)


@dataclass
class DebiasConfig:
    """Task 1 (insid3_ext/debiasing.py). Formulas: docs/MATHEMATICAL_FORMULATION.md section 2."""
    mode: str = "hard"                  # official default. Our modes: none | weighted | spectral | calibrated
                                        # | image | token; controls: random | pca | rho_threshold
    rank: Optional[int] = None          # how many directions to act on; None = the mode's natural choice
    alpha: float = 1.0                  # weighted: fraction removed. 1.0 = official (so the default changes nothing)
    knee_rank: int = 500                # spectral: where lambda = 0.5; 500 = the official cut-off point
    sharpness: float = 1.0              # spectral: 1 = classical Wiener curve; large = hard cut (swept 1, 4)
    strength: float = 1.0               # beta for calibrated/image/token: 1 = the MMSE optimum (swept)
    calibration_path: Optional[str] = None   # file from scripts/calibrate_positional.py
    variance_source: str = "calibration"     # image mode: positional energy measured on real images | "noise" (empty image)
    template_source: str = "calibration"     # token mode: pattern measured on real images | "noise" (empty image)
    token_gain: str = "least_squares"        # token mode: fit the pattern's strength per patch | "fixed" (= beta)
    max_gain: float = 2.0               # token mode: cap on the fitted strength (avoids over-subtracting
                                        # where the fit is unstable); starting value
    rho_threshold: float = 0.3          # control: remove directions with measured positional share >= 0.3
                                        # (calibration: share ~0.9 for the first few, ~0.05 noise floor later;
                                        # 0.3 lies clearly between); swept 0.2 / 0.3 / 0.5
    control_seed: int = 0               # control random: which random directions (reproducible)
    center: bool = False                # control: subtract the mean real-image feature first
                                        # (DINOv3 report App. A.2: some feature dimensions are constant
                                        # across all patches and images)


@dataclass
class PrototypeConfig:
    """Task 2 (insid3_ext/prototypes.py, matching.py). Formulas: docs/MATHEMATICAL_FORMULATION.md section 3."""
    enabled: bool = False               # False = official single prototype
    num_prototypes: int = 4             # M: starting value; swept 1..8 (ASGNet found ~5 best, with training)
    clustering: str = "kmeans"          # kmeans (fixed M) | agglomerative (M from the data, like INSID3's own clustering)
    cluster_space: str = "original"     # group with ORIGINAL features, as INSID3 groups the target | "debiased"
    ref_tau: float = 0.6                # agglomerative threshold; 0.6 = INSID3's tau for the target
    kmeans_iters: int = 50              # Lloyd iterations; k-means on <= 4096 points converges well before 50
    kmeans_restarts: int = 1            # k-means runs with different starts (best kept)
    min_points_per_prototype: int = 4   # a group needs at least this many patches; small masks get fewer
                                        # groups. 4 = minimal; 100 = ASGNet's area rule (swept)
    min_cluster_fraction: float = 0.0   # optionally drop groups smaller than this share of the mask (off)
    include_global: bool = False        # optionally also keep the original prototype
    global_weight: float = 0.5          # its weight if kept (half: equal footing with the parts)
    token_aggregation: str = "max"      # how M similarities become one: single | max | weighted | lse
    lse_temperature: float = 0.05       # lse "softness": cosine scores differ by ~0.05-0.2 between parts,
                                        # so T = 0.05 sits between max and average; swept 0.02 / 0.05 / 0.1
    region_matching: str = "prototype"  # region score: prototype (as INSID3) | chamfer | ot
    ot_epsilon: float = 0.05            # OT entropy: same scale as lse_temperature; swept 0.02 / 0.05 / 0.1
    ot_rho: float = 0.5                 # OT: how strongly a cluster must cover all parts (0 = not at all,
                                        # large = completely); swept 0.1 .. 100
    ot_iters: int = 50                  # Sinkhorn iterations (enough for these small problems)
    seed: int = 0                       # seed of k-means' own random generator (never the global one)
    multishot: str = "per_shot"         # several shots: cluster each shot | "pooled" (all shots together)
    score_calibration: str = "none"     # "reference": rescale scores to the original's scale (see model.py)


@dataclass
class DataConfig:
    dataset: str = "coco"
    data_root: str = "data"
    fold: int = 0
    shots: int = 1                      # official main setting: 1-shot
    split: str = "test"                 # test = official episodes | tune = COCO train2014 episodes (model selection)
    tune_episodes: int = 1000           # round 1 used 1000; round 2 uses 4000 (1000 is too noisy, PFENet Tab. 10)
    num_workers: int = 0                # must stay 0: the official datasets draw episodes from the global
                                        # numpy random generator; worker processes would change them


@dataclass
class RunConfig:
    output_root: str = "outputs"
    exp_name: str = "baseline"
    seed: int = 0                       # official seed
    max_episodes: Optional[int] = None  # only for quick tests: stop after this many episodes
    resume: bool = True                 # skip episodes already in episodes.jsonl
    save_predictions: int = 0           # save the first N predicted masks as images
    log_every: int = 50                 # print the running mIoU every N episodes
    full_diagnostics: bool = False      # log candidates / seed / scores also for baseline runs (same result)


@dataclass
class CorrespondenceConfig:
    debiased: bool = True               # False = original DINOv3 features (the "no debiasing" reference)
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
DEBIAS_MODES = ("none", "hard", "weighted", "spectral", "calibrated", "image", "token",
                "random", "pca", "rho_threshold")
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
    return (d.center or d.mode in ("calibrated", "pca", "rho_threshold")
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
    _check(0.0 <= d.rho_threshold <= 1.0, "debias.rho_threshold must lie in [0, 1]")
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
    _check(p.multishot in ("per_shot", "pooled"), "prototypes.multishot: per_shot | pooled")
    _check(p.score_calibration in ("none", "reference"), "prototypes.score_calibration: none | reference")
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
