"""``INSID3Ext`` — the official INSID3 model with Task 1 and Task 2 plugged in.

Inheritance, not modification: ``INSID3Ext`` subclasses ``models.insid3.INSID3``.

* Task 1 only: ``_debias_features`` is overridden. Every consumer of debiased features
  in the official ``predict_mask`` / ``predict_keypoint`` (prototype, forward mask,
  backward NN, cluster prototypes, cross score) then uses the adaptive version.
  With ``debias.mode='hard'`` the *parent* method is called, i.e. the exact original.
* Task 2: ``predict_mask`` is overridden only when ``prototypes.enabled``. The override
  is a copy of the official method in which the three uses of the single prototype
  (forward mask, seed selection, cross score in aggregation) are replaced by the
  multi-prototype equivalents. Backward correspondence, clustering, intra-image
  similarity, area weighting and thresholding are unchanged (copied verbatim).
  With M = 1 and region_matching='prototype' the override reproduces the original
  for 1-shot episodes (tests/test_baseline_equivalence.py).

Shapes inside predict_mask (S shots, C channels, h×w token grid, P = h*w)
    imgs            (1, S+1, 3, H, W)
    fmaps_norm      (1, S+1, C, h, w)
    feat_tgt(_deb)  (1, C, h, w)
    token_sims      (P, M)
    token_score     (h, w)
    cluster_labels  (h, w), K clusters
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from models.insid3 import INSID3
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import downsample_mask

from .config import DebiasConfig, PrototypeConfig, needs_calibration
from .debiasing import PositionalDebiaser, estimate_positional_svd, load_calibration
from .matching import (aggregate_similarities, region_scores_chamfer, region_scores_ot,
                       segment_mean)
from .prototypes import build_reference_prototypes


class INSID3Ext(INSID3):
    """INSID3 with adaptive positional debiasing and multi-prototype matching."""

    def __init__(self, encoder: torch.nn.Module, *, debias_cfg: Optional[DebiasConfig] = None,
                 proto_cfg: Optional[PrototypeConfig] = None, model_size: Optional[str] = None,
                 profile_stages: bool = False, **insid3_kwargs: Any):
        """
        Args:
            encoder: frozen DINOv3 backbone (must provide ``get_intermediate_layers``).
            debias_cfg: Task 1 configuration (default: official hard projection).
            proto_cfg: Task 2 configuration (default: disabled = official single prototype).
            model_size: used only to validate a calibration file.
            profile_stages: record per-stage wall-clock times (adds device syncs).
            **insid3_kwargs: forwarded unchanged to ``INSID3.__init__``.
        """
        self.debias_cfg = debias_cfg or DebiasConfig()
        self.proto_cfg = proto_cfg or PrototypeConfig()
        self._pos_svd = None
        super().__init__(encoder, **insid3_kwargs)   # calls our _build_positional_basis

        calibration = None
        if self.debias_cfg.calibration_path and needs_calibration(self.debias_cfg):
            calibration = load_calibration(self.debias_cfg.calibration_path,
                                           model_size=model_size, image_size=self.image_size)
        self.debiaser = PositionalDebiaser.from_config(
            self.debias_cfg, self._pos_svd, self.svd_components, calibration)
        self.profile_stages = profile_stages
        self.last_diagnostics: Dict[str, Any] = {}

    # ──────── Task 1 hooks ────────

    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Same basis as the original, but the full SVD is kept for the soft modes."""
        self._pos_svd = estimate_positional_svd(self.encoder, self.image_size, device)
        return self._pos_svd.U[:, :self.svd_components].contiguous()

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """(B, T, C, h, w) → (B, T, C, h, w). 'hard' delegates to the official method."""
        if self.debias_cfg.mode == "hard":
            return super()._debias_features(fmaps_norm)
        return self.debiaser(fmaps_norm)

    # ──────── profiling helper ────────

    @contextmanager
    def _stage(self, name: str):
        if not self.profile_stages:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_diagnostics.setdefault("stage_ms", {})[name] = 1000 * (time.perf_counter() - t0)

    # ──────── Task 2: segmentation ────────

    @torch.no_grad()
    def predict_mask(self, ref_images: torch.Tensor, ref_masks: torch.Tensor,
                     tgt_image: torch.Tensor) -> torch.Tensor:
        """Segment the target given S references.

        Args:
            ref_images: (S, 3, H, W); ref_masks: (S, H, W) bool; tgt_image: (1, 3, H, W).
        Returns:
            (H', W') bool mask (see ``_finalize_mask``).
        """
        self.last_diagnostics = {}
        if not self.proto_cfg.enabled:
            with self._stage("total"):
                return super().predict_mask(ref_images, ref_masks, tgt_image)
        cfg = self.proto_cfg

        S = ref_images.shape[0]
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)

        with self._stage("encoder"):
            fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, C, h, w = fmaps_norm.shape
        ref_masks = ref_masks.unsqueeze(1)                                  # (S, 1, H, W)
        feat_tgt = fmaps_norm[:, S]

        with self._stage("debias"):
            fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]

        # ---- NEW: multi-prototype reference representation ----
        with self._stage("prototypes"):
            group_feats = fmaps_norm[0, :S] if cfg.cluster_space == "original" else feat_refs_deb[0]
            pset = build_reference_prototypes(feat_refs_deb[0], group_feats, ref_masks, cfg)
        Q = pset.prototypes                                                 # (M, C)
        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)    # (P, C)
        token_sims = feat_tgt_deb_flat.float() @ Q.T.float()                # (P, M), fp32 scoring
        if cfg.token_aggregation == "single":
            gp = pset.global_prototype.float()
            token_score = feat_tgt_deb_flat.float() @ gp                     # (P,)
        else:
            token_score = aggregate_similarities(token_sims, pset.weights, cfg.token_aggregation,
                                                 cfg.lse_temperature)
        token_score_map = token_score.view(h, w)

        # ---- Candidate localization (forward: NEW score; backward: official) ----
        with self._stage("candidates"):
            sim_maps = []
            for m in range(S):
                feat_ref_m = feat_refs_deb[:, m]
                sim_maps.append(torch.einsum('bchw,bcxy->bhwxy', feat_ref_m, feat_tgt_deb))
            forward_mask = token_score_map > 0
            if forward_mask.sum() == 0:
                forward_mask = token_score_map > float(torch.quantile(token_score_map.float(), 0.9))
            candidate_mask = forward_mask & self._backward_mask(sim_maps, ref_masks, h, w)
        self.last_diagnostics.update(num_prototypes=pset.num,
                                     candidate_tokens=int(candidate_mask.sum()))
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        # ---- Fine-grained clustering (official) ----
        with self._stage("clustering"):
            feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
            cluster_labels = agglomerative_clustering(feat_tgt_flat, self.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1
        cluster_protos = compute_cluster_prototypes(feat_tgt_deb_flat, cluster_labels.view(-1), K)

        with self._stage("aggregation"):
            seed_scores, cross_sim = self._region_scores(
                token_sims, token_score, cluster_protos, cluster_labels.view(-1), K, pset)
            pred_mask = self._seed_and_aggregate_scores(
                candidate_mask, cluster_labels, K, seed_scores, cross_sim, feat_tgt, h, w)
        self.last_diagnostics.update(num_clusters=K)
        return self._finalize_mask(pred_mask, tgt_image)

    def _backward_mask(self, sim_maps: list, ref_masks: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Majority-vote backward NN filter (verbatim from ``INSID3._locate_candidates``)."""
        import math
        k = len(sim_maps)
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]                                         # (Hs, Ws, h, w)
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)                    # (h, w, Hs, Ws)
            best_idx = sim_t_to_r.reshape(h, w, -1).argmax(dim=2)    # (h, w)
            rows = best_idx // Ws
            cols = best_idx % Ws
            ref_mask_m = downsample_mask(ref_masks[m:m + 1], Hs, Ws).squeeze(0)
            votes += ref_mask_m[rows, cols].to(torch.int32)
        return votes >= math.ceil(k / 2)

    def _region_scores(self, token_sims, token_score, cluster_protos, labels, K, pset):
        """Return (seed_scores (K,), cross_sim (K,)).

        'prototype' mirrors the original: seed uses cluster prototype ↔ reference
        prototype(s); aggregation uses the mean token score inside each cluster.
        """
        cfg = self.proto_cfg
        if cfg.region_matching == "prototype":
            cp = cluster_protos.float()
            if cfg.token_aggregation == "single":
                seed = cp @ pset.global_prototype.float()                    # (K,)
            else:
                proto_sims = cp @ pset.prototypes.T.float()                  # (K, M)
                seed = aggregate_similarities(proto_sims, pset.weights, cfg.token_aggregation,
                                              cfg.lse_temperature)
            cross = segment_mean(token_score, labels, K)
            return seed, cross
        if cfg.region_matching == "chamfer":
            r = region_scores_chamfer(token_sims, labels, K, pset.weights)
        else:
            r = region_scores_ot(token_sims, labels, K, pset.weights,
                                 cfg.ot_epsilon, cfg.ot_rho, cfg.ot_iters)
        return r, r

    def _seed_and_aggregate_scores(self, candidate_mask, cluster_labels, K, seed_scores,
                                   cross_sim, feat_tgt, h, w) -> torch.Tensor:
        """Official ``_seed_and_aggregate`` with the two cross-image scores passed in."""
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask
        matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

        all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
        per_cluster = torch.zeros(K, device=cluster_labels.device)
        per_cluster[matched_ids] = n_pixels.float()
        area_weights = per_cluster / all_areas

        seed_idx = int(torch.argmax(seed_scores[matched_ids]).item())
        seed_id = matched_ids[seed_idx].item()

        feat_tgt_flat = feat_tgt[0].reshape(feat_tgt.shape[1], -1).permute(1, 0)
        orig_protos = compute_cluster_prototypes(feat_tgt_flat, cluster_labels.view(-1), K)
        intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)

        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined *= area_weights

        self.last_diagnostics.update(seed_cluster=int(seed_id),
                                     seed_score=float(seed_scores[seed_id]))
        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
        return final_mask
