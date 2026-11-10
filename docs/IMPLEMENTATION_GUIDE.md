# Implementation guide

## 1. Layout

```
INSID3_Research/
├── models/ utils/ datasets/ opts.py inference_*.py   OFFICIAL INSID3, byte-identical (verify_upstream.py)
├── upstream/            UPSTREAM_COMMIT, UPSTREAM_MANIFEST.sha256, UPSTREAM_README.md
├── insid3_ext/          ALL NEW CODE
│   ├── config.py        dataclass configs, YAML _base_ inheritance, ${env:VAR,default}, --set overrides, validation
│   ├── debiasing.py     Task 1: estimate_positional_svd, gain schedules, PositionalDebiaser
│   ├── calibration.py   Task 1: PositionalStatsAccumulator (ρ, templates, per-image/token stats)
│   ├── prototypes.py    Task 2: spherical_kmeans, build_reference_prototypes, PrototypeSet
│   ├── matching.py      Task 2: aggregation rules, segment reductions, Chamfer, unbalanced Sinkhorn
│   ├── model.py         INSID3Ext(INSID3): wiring of both tasks
│   ├── builder.py       DINOv3 loading with env-configurable paths, build_model(cfg)
│   ├── datasets_ext.py  official builders + COCO tuning split
│   ├── experiment.py    resumable segmentation / correspondence loops
│   └── runtime.py       CUDA checks, seeding, timing, peak memory, environment report
├── configs/             base.yaml, baseline*.yaml, task1/, task2/, combined/, correspondence/, selected/ (generated)
├── scripts/             CLI entry points (run_*, calibrate, select_best, aggregate, compare*, analyze, visualize, check_env)
├── tests/               CPU test suite (fake encoder)
├── *.sh                 server workflow
└── docs/                documentation
```

## 2. Design decisions

1. **Subclass, never edit.** `INSID3Ext` inherits `INSID3`. With `debias.mode=hard` the parent's
   `_debias_features` runs; with `prototypes.enabled=false` the parent's `predict_mask` runs. Hence the
   baseline config executes *exactly* the official code (plus our logging around it).
2. **One override per task.** Task 1 = `_build_positional_basis` + `_debias_features`. Task 2 =
   `predict_mask` (a copy of the official method where only the three prototype uses changed; the copied
   blocks are marked "official"/"verbatim").
3. **The basis is recomputed identically.** `estimate_positional_svd` repeats the official operations and
   additionally keeps S and the template coefficients; `U[:, :s]` is the official basis (tested with
   `torch.equal`).
4. **RNG isolation.** Episodes come from the global numpy RNG, so new code must not consume it. k-means
   uses `torch.Generator(seed)`; calibration image sampling uses `random.Random(seed)`. Test:
   `test_kmeans_deterministic_and_rng_isolated`; end-to-end: `test_methods_see_identical_episodes`.
5. **Precision.** Debiasing runs in the feature dtype (as the official code); data statistics (image-mode
   second moments, token LS gains, calibration sums) in fp32/fp64.
6. **No hidden state between episodes.** The model caches nothing across episodes except the basis and
   (optional) calibration; `segment()` resets the reference/target state as in the original.
7. **Fail early.** Config validation (`ConfigError`) and CUDA/weights checks happen before any heavy work.

## 3. Key functions (inputs → outputs)

| Function | Input | Output |
|---|---|---|
| `estimate_positional_svd(encoder, image_size, device)` | encoder | `PositionalSVD(U (C,R), S (R,), coeff (R,P), grid)` |
| `PositionalDebiaser.from_config(cfg, svd, s, calibration)` | DebiasConfig | debiaser |
| `PositionalDebiaser.__call__(fmaps)` | (B,T,C,h,w) unit-norm | (B,T,C,h,w) unit-norm |
| `PositionalStatsAccumulator.update(feats)` / `.finalize()` | (N,C,h,w) | dict: U,S,rho,pos_second_moment,template,… |
| `spherical_kmeans(X, k, iters, gen)` | (N,C) unit-norm | labels (N,), centres (k,C), objective |
| `build_reference_prototypes(f_match, f_group, masks, cfg)` | (S,C,h,w)×2, (S,1,H,W) | `PrototypeSet` |
| `aggregate_similarities(sims, w, mode, T)` | (…,M) | (…) |
| `region_scores_chamfer / region_scores_ot(sims, labels, K, w, …)` | (P,M), (P,) | (K,) |
| `INSID3Ext.predict_mask(ref_imgs, ref_masks, tgt)` | (S,3,H,W), (S,H,W), (1,3,H,W) | (H',W') bool |
| `run_segmentation(cfg, model)` | config | metrics dict; writes run directory |

`INSID3Ext.last_diagnostics` (logged per episode): `num_prototypes`, `candidate_tokens`, `num_clusters`,
`seed_cluster`, `seed_score`, `seed_cross`, `forward_tokens`, `backward_tokens`, `selected_clusters`.

## 4. Extending

* New debias mode: add the name to `DEBIAS_MODES` (config.py), build its λ in
  `PositionalDebiaser.from_config`, apply it in `__call__`, add tests in `tests/test_debiasing.py`.
* New aggregation / region rule: `TOKEN_AGGREGATIONS` / `REGION_MATCHINGS` in config.py, implement in
  matching.py, dispatch in `INSID3Ext._region_scores`.
* New experiment: a YAML with `_base_: ../base.yaml` and only the changed keys, or `--set` overrides.

## 5. Running the tests

```bash
python -m pytest tests -q
```

The fake encoder (tests/conftest.py) is a 16×16-patch conv with an injected low-rank positional field,
8×8 tokens and 32 channels. It checks shapes, code paths, equivalences and invariants — not accuracy.

## 6. Revision after the first experimental round

| Addition | Where | Purpose |
|---|---|---|
| `debias.mode=random` | debiasing.py | hard projection of a random s-dim subspace (control) |
| `debias.mode=pca` | debiasing.py, calibration.py (`feat_mean`, `feat_cov`) | hard projection of the top-s natural-image PCA directions (control) |
| `debias.mode=rho_threshold` | debiasing.py | remove only components with ρ_k ≥ τ |
| `PositionalDebiaser(..., normalize=False)` | debiasing.py | pre-normalisation output for energy measurements |
| `prototypes.multishot=pooled` | prototypes.py (`_build_pooled`, weighted k-means) | multi-shot prototypes with original shot weighting; M = 1 ≡ INSID3 for any S |
| `prototypes.score_calibration=reference` | model.py (`_score_scale`) | label-free rescaling of multi-prototype scores to the single-prototype scale |
| `run.full_diagnostics=true` | model.py, builder.py | identical predictions, but candidates / seed / selected clusters / scores logged for baseline and Task-1 runs |
| correspondence statistics | compare_all.py, compare_runs.py | paired bootstrap per PCK threshold vs the `hard` run of the same backbone |
| seed averaging, combined from `c_` runs | select_best.py | multi-seed tuning; uses the combined stage's own winner |
| `scripts/analyze_positional_bias.py` | new | residual positional bias per mode on held-out images |
| `run_revision.sh`, `scripts/check_consistency.py` | new | phased revision experiments; same-configuration reproducibility check |
| `debias.center` | model.py, config.py | subtract the natural-image mean feature before debiasing (control, DINOv3 App. A.2) |
