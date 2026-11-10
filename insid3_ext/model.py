"""INSID3Ext = the official INSID3 model with our two changes plugged in.

We never edit the official file models/insid3.py. Instead INSID3Ext INHERITS from the official class
INSID3 and replaces only the methods we need to change:

  Task 1  _debias_features        -> uses our PositionalDebiaser (debiasing.py)
          _build_positional_basis -> same basis as before, but keeps the full SVD for the new modes
  Task 2  predict_mask            -> a copy of the official predict_mask in which ONLY the three places
                                     that use the single reference prototype are changed:
                                       (a) forward filter, (b) choice of the seed cluster,
                                       (c) cross-image score when merging clusters.
                                     Everything else (backward matching, clustering, intra-image
                                     similarity, area weights, threshold) is copied unchanged.

When nothing is switched on (debias.mode = hard, prototypes.enabled = false) the official methods run
themselves, so the baseline is exactly INSID3. Tests prove the equivalences:
tests/test_baseline_equivalence.py and tests/test_revision.py.

Shapes inside predict_mask (S shots, C channels, h x w patches, P = h*w)
  imgs            (1, S+1, 3, H, W)   the S reference images and the target
  fmaps_norm      (1, S+1, C, h, w)   unit-length features
  token_sims      (P, M)              each target patch vs each prototype
  token_score     (P,)                one score per target patch
  cluster_labels  (h, w)              target clusters 0..K-1
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from models.insid3 import INSID3
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import downsample_mask

from .config import DebiasConfig, PrototypeConfig, needs_calibration
from .debiasing import PositionalDebiaser, estimate_positional_svd, load_calibration
from .matching import aggregate_similarities, region_scores_chamfer, region_scores_ot, segment_mean
from .prototypes import build_reference_prototypes


class INSID3Ext(INSID3):
    """INSID3 with adaptive positional debiasing (Task 1) and multi-prototype matching (Task 2)."""

    def __init__(self, encoder: torch.nn.Module, *, debias_cfg: Optional[DebiasConfig] = None,
                 proto_cfg: Optional[PrototypeConfig] = None, model_size: Optional[str] = None,
                 full_diagnostics: bool = False, basis_eval_mode: bool = False, **insid3_kwargs: Any):
        """
        encoder           frozen DINOv3 backbone
        debias_cfg        Task-1 settings (default = official hard projection)
        proto_cfg         Task-2 settings (default = off = official single prototype)
        model_size        only used to check that a calibration file belongs to this backbone
        full_diagnostics  also log candidates / seed / scores when Task 2 is off (see predict_mask)
        basis_eval_mode   compute the positional basis with the encoder in eval mode (see _build_positional_basis)
        insid3_kwargs     passed unchanged to the official INSID3 constructor
        """
        self.debias_cfg = debias_cfg or DebiasConfig()
        self.proto_cfg = proto_cfg or PrototypeConfig()
        self._pos_svd = None
        self.basis_eval_mode = basis_eval_mode       # must exist before the official constructor runs
        # The official constructor computes the positional basis by calling self._build_positional_basis,
        # which is OUR version below; it stores the full SVD in self._pos_svd.
        super().__init__(encoder, **insid3_kwargs)

        # Load the calibration file only if the chosen mode needs it.
        calibration = None
        if self.debias_cfg.calibration_path and needs_calibration(self.debias_cfg):
            calibration = load_calibration(self.debias_cfg.calibration_path,
                                           model_size=model_size, image_size=self.image_size)
        self.debiaser = PositionalDebiaser.from_config(self.debias_cfg, self._pos_svd,
                                                       self.svd_components, calibration)

        # Optional control 'center': the mean feature of real images, subtracted before debiasing.
        self._center = None
        if self.debias_cfg.center:
            if "feat_mean" not in calibration:
                raise ValueError("debias.center needs a calibration file with 'feat_mean' "
                                 "(re-run calibration with the revised code).")
            self._center = calibration["feat_mean"].float()

        # With Task 2 off, full_diagnostics runs our predict_mask with the ORIGINAL single prototype
        # (token_aggregation 'single', M = 1). The prediction is identical to the official one (tested),
        # but we can log the intermediate numbers, so baseline and methods can be compared episode by episode.
        self.full_diagnostics = full_diagnostics
        self._diag_cfg = PrototypeConfig(enabled=True, num_prototypes=1, token_aggregation="single",
                                         region_matching="prototype")
        self.last_diagnostics: Dict[str, Any] = {}

    # ════════════════════════════════════════════════════════════════════════════════════
    # Task 1
    # ════════════════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _build_positional_basis(self, device: str) -> torch.Tensor:
        """Replaces the official method: same basis U[:, :s], but we keep the whole SVD."""
        # ORIGINAL (models/insid3.py L246-261), last three lines:
        #     E = E - E.mean(dim=1, keepdim=True)
        #     U, _, _ = torch.linalg.svd(E, full_matrices=False)
        #     return U[:, :self.svd_components].contiguous()
        # OURS: the same computation (estimate_positional_svd repeats L248-259 line by line), but we also
        # keep the singular values S and all directions, which the new debiasing modes need.
        #
        # Finding (2026-11-10): this runs inside the constructor, i.e. BEFORE the inference script calls
        # model.eval() (inference_segmentation.py L44-45). The DINOv3 hub model is still in TRAINING mode,
        # and DINOv3's RoPE then multiplies all patch coordinates by a random factor in [1/2, 2]
        # (pos_embed_rope_rescale_coords=2; dinov3/layers/rope_position_encoding.py). So the official basis
        # comes from one randomly rescaled forward pass (reproducible only through the seed), while every real
        # feature is computed later in eval mode without rescaling.
        # basis_eval_mode = False keeps the official behaviour; True computes the basis in eval mode.
        if self.basis_eval_mode:
            was_training = self.encoder.training
            self.encoder.eval()
            self._pos_svd = estimate_positional_svd(self.encoder, self.image_size, device)
            self.encoder.train(was_training)
        else:
            self._pos_svd = estimate_positional_svd(self.encoder, self.image_size, device)
        return self._pos_svd.U[:, :self.svd_components].contiguous()

    def _debias_features(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """(B, T, C, h, w) -> (B, T, C, h, w). Used for all cross-image comparisons.

        Only cross-image features are debiased; clustering of the target and the intra-image similarity
        keep using the original features, as in INSID3.
        """
        # ORIGINAL (models/insid3.py L263-271):
        #     B, T, C, H, W = fmaps_norm.shape
        #     X = fmaps_norm.reshape(B * T, C, H * W)
        #     basis = self.positional_basis.to(device=X.device, dtype=X.dtype)
        #     P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
        #     X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
        #     return F.normalize(X_deb, p=2, dim=2)
        # OURS: mode 'hard' calls exactly that original method (super()); every other mode goes
        # through PositionalDebiaser. Optional 'center' control first.
        if self._center is not None:
            # Control 'center': x <- normalize(x - mean feature of real images).
            mu = self._center.to(device=fmaps_norm.device, dtype=fmaps_norm.dtype).view(1, 1, -1, 1, 1)
            fmaps_norm = F.normalize(fmaps_norm - mu, p=2, dim=2)
        if self.debias_cfg.mode == "hard":
            return super()._debias_features(fmaps_norm)       # the official code itself
        return self.debiaser(fmaps_norm)

    # ════════════════════════════════════════════════════════════════════════════════════
    # Task 2 — the official predict_mask (models/insid3.py lines 121-189) with three changes
    # ════════════════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def predict_mask(self, ref_images: torch.Tensor, ref_masks: torch.Tensor,
                     tgt_image: torch.Tensor) -> torch.Tensor:
        """ref_images (S, 3, H, W), ref_masks (S, H, W) bool, tgt_image (1, 3, H, W) -> mask."""
        self.last_diagnostics = {}
        if not self.proto_cfg.enabled and not self.full_diagnostics:
            return super().predict_mask(ref_images, ref_masks, tgt_image)    # official, untouched
        cfg = self.proto_cfg if self.proto_cfg.enabled else self._diag_cfg

        # ── Official step 1: features of references + target, unit length (lines 133-142) ──
        S = ref_images.shape[0]
        imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)
        fmaps = self._extract_features(imgs)
        fmaps_norm = F.normalize(fmaps, p=2, dim=2)
        _, _, C, h, w = fmaps_norm.shape
        ref_masks = ref_masks.unsqueeze(1)                                   # (S, 1, H, W)
        feat_tgt = fmaps_norm[:, S]                                          # original target features

        # ── Official step 2: debiased features for cross-image matching (lines 144-147) ──
        fmaps_debiased = self._debias_features(fmaps_norm)
        feat_refs_deb = fmaps_debiased[:, :S]
        feat_tgt_deb = fmaps_debiased[:, S]

        # ── CHANGED: the reference prototype ──
        # ORIGINAL (models/insid3.py L149-158):
        #     ref_prototypes = []
        #     for s in range(S):
        #         mask_s = downsample_mask(ref_masks[s:s+1], h, w)
        #         fg = feat_refs_deb[0, s, :, mask_s]
        #         if fg.shape[1] > 0:
        #             ref_prototypes.append(fg.mean(dim=1))
        #     ref_prototype = F.normalize(
        #         torch.stack(ref_prototypes).mean(dim=0), p=2, dim=0
        #     ).unsqueeze(1)
        # OURS: a SET of prototypes (prototypes.py). The original single prototype is still computed
        # inside it (pset.global_prototype, same expression) and used when token_aggregation = 'single'.
        group_feats = fmaps_norm[0, :S] if cfg.cluster_space == "original" else feat_refs_deb[0]
        pset = build_reference_prototypes(feat_refs_deb[0], group_feats, ref_masks, cfg)
        kappa = self._score_scale(feat_refs_deb[0], ref_masks, pset, cfg)    # 1.0 unless calibrated

        # One score per target patch (fp32 for accuracy).
        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)     # (P, C)
        token_sims = feat_tgt_deb_flat.float() @ pset.prototypes.T.float()   # (P, M)
        if cfg.token_aggregation == "single":
            token_score = feat_tgt_deb_flat.float() @ pset.global_prototype.float()   # original <x, p>
        else:
            token_score = aggregate_similarities(token_sims, pset.weights, cfg.token_aggregation,
                                                 cfg.lse_temperature)
        token_score_map = token_score.view(h, w)

        # ── Candidate patches (official _locate_candidates, lines 275-307) ──
        # Dense similarity of every reference patch to every target patch (needed for backward matching).
        sim_maps = []
        for m in range(S):
            sim_maps.append(torch.einsum('bchw,bcxy->bhwxy', feat_refs_deb[:, m], feat_tgt_deb))
        # (a) CHANGED: forward filter.
        # ORIGINAL (models/insid3.py L286-289):
        #     sim_fwd = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
        #     forward_mask = sim_fwd > 0
        #     if forward_mask.sum() == 0:
        #         forward_mask = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
        # OURS: the same rule (score > 0, else top 10 %), with our patch score instead of <x, p>.
        forward_mask = token_score_map > 0
        if forward_mask.sum() == 0:
            forward_mask = token_score_map > float(torch.quantile(token_score_map.float(), 0.9))
        # Backward filter: unchanged official code.
        backward_mask = self._backward_mask(sim_maps, ref_masks, h, w)
        candidate_mask = forward_mask & backward_mask
        self.last_diagnostics.update(num_prototypes=pset.num,
                                     forward_tokens=int(forward_mask.sum()),
                                     backward_tokens=int(backward_mask.sum()),
                                     candidate_tokens=int(candidate_mask.sum()),
                                     score_scale=float(kappa))
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        # ── Official step: cluster the target with ORIGINAL features (lines 173-181) ──
        feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
        cluster_labels = agglomerative_clustering(feat_tgt_flat, self.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1
        cluster_protos = compute_cluster_prototypes(feat_tgt_deb_flat, cluster_labels.view(-1), K)

        # ── (b) and (c) CHANGED: the two cross-image scores of every cluster ──
        seed_scores, cross_sim = self._region_scores(token_sims, token_score, cluster_protos,
                                                     cluster_labels.view(-1), K, pset, cfg)
        cross_sim = cross_sim * kappa          # kappa = 1 unless score calibration is on
        # ── Seed choice + merging: official code with our two scores (lines 311-364) ──
        pred_mask = self._seed_and_aggregate_scores(candidate_mask, cluster_labels, K, seed_scores,
                                                    cross_sim, feat_tgt, h, w)
        self.last_diagnostics.update(num_clusters=K)
        return self._finalize_mask(pred_mask, tgt_image)

    def _backward_mask(self, sim_maps: list, ref_masks: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Official backward filter (lines 292-305), unchanged.

        For every target patch, find its most similar patch in each reference image; it is a candidate
        if that nearest neighbour lies inside the reference mask for a majority of the references.
        """
        k = len(sim_maps)
        votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
        for m, sim_m in enumerate(sim_maps):
            sim0 = sim_m[0]                                          # (Hs, Ws, h, w)
            Hs, Ws = sim0.shape[:2]
            sim_t_to_r = sim0.permute(2, 3, 0, 1)                     # (h, w, Hs, Ws)
            best_idx = sim_t_to_r.reshape(h, w, -1).argmax(dim=2)     # nearest reference patch
            rows = best_idx // Ws
            cols = best_idx % Ws
            ref_mask_m = downsample_mask(ref_masks[m:m + 1], Hs, Ws).squeeze(0)
            votes += ref_mask_m[rows, cols].to(torch.int32)          # +1 if it is inside the mask
        return votes >= math.ceil(k / 2)

    def _score_scale(self, feat_refs_deb, ref_masks, pset, cfg) -> float:
        """Optional score calibration kappa (only with prototypes.score_calibration = 'reference').

        Multi-prototype scores can be on a different scale than the original single-prototype score,
        while the merge threshold alpha = 0.2 was chosen for the original scale. On the REFERENCE patches
        (no target labels involved) we compare the two scores and rescale:
            kappa = mean <x, p_original> / mean multi-prototype score
        """
        if cfg.score_calibration != "reference" or cfg.token_aggregation == "single":
            return 1.0
        S, C, h, w = feat_refs_deb.shape
        toks = [feat_refs_deb[s, :, downsample_mask(ref_masks[s:s + 1], h, w)].T for s in range(S)]
        X = torch.cat(toks).float()                                          # all masked reference patches
        single = (X @ pset.global_prototype.float()).mean()
        agg = aggregate_similarities(X @ pset.prototypes.T.float(), pset.weights,
                                     cfg.token_aggregation, cfg.lse_temperature).mean()
        return float(single / agg) if float(agg) > 1e-6 else 1.0

    def _region_scores(self, token_sims, token_score, cluster_protos, labels, K, pset, cfg):
        """The two cross-image scores of each target cluster: (seed_scores (K,), cross_sim (K,)).

        region_matching = 'prototype' follows the official code:
            seed score = similarity of the cluster's prototype to the reference prototype(s)  (line 338)
            cross score = mean patch score inside the cluster                              (lines 350-354)
        'chamfer' / 'ot' use one set-to-set score (matching.py) for both.
        """
        # ORIGINAL seed score (models/insid3.py L337-338):
        #     protos_matched = cluster_protos[matched_ids]
        #     cross_sim_matched = (ref_prototype.T @ protos_matched.T).squeeze(0)
        # ORIGINAL cross score (models/insid3.py L350-354):
        #     fg_sim = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
        #     cross_sim = torch.empty(K, device=fg_sim.device, dtype=fg_sim.dtype)
        #     for k in range(K):
        #         idx = (cluster_labels == k)
        #         cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0
        # OURS ('prototype'): seed = cluster prototype vs the prototype SET (aggregated), cross = mean of our
        # patch score per cluster (segment_mean = the same loop, done for all clusters at once).
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
            r = region_scores_ot(token_sims, labels, K, pset.weights, cfg.ot_epsilon, cfg.ot_rho, cfg.ot_iters)
        return r, r

    def _seed_and_aggregate_scores(self, candidate_mask, cluster_labels, K, seed_scores,
                                   cross_sim, feat_tgt, h, w) -> torch.Tensor:
        """Official _seed_and_aggregate (lines 311-364); only seed_scores / cross_sim come from outside."""
        # Clusters that contain at least one candidate patch.
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask
        matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

        # Area weight = fraction of each cluster that is candidate (0 for clusters with no candidate).
        all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
        per_cluster = torch.zeros(K, device=cluster_labels.device)
        per_cluster[matched_ids] = n_pixels.float()
        area_weights = per_cluster / all_areas

        # Seed = matched cluster with the highest seed score.
        # ORIGINAL (L339): seed_idx = int(torch.argmax(cross_sim_matched).item())
        # OURS: the same argmax over the matched clusters, of the score computed in _region_scores.
        seed_idx = int(torch.argmax(seed_scores[matched_ids]).item())
        seed_id = matched_ids[seed_idx].item()

        # Intra-image similarity of every cluster to the seed, with ORIGINAL features.
        feat_tgt_flat = feat_tgt[0].reshape(feat_tgt.shape[1], -1).permute(1, 0)
        orig_protos = compute_cluster_prototypes(feat_tgt_flat, cluster_labels.view(-1), K)
        intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)

        # Final score = cross-image score x intra-image score x area weight (seed's weight forced to 1).
        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined *= area_weights

        self.last_diagnostics.update(seed_cluster=int(seed_id),
                                     seed_score=float(seed_scores[seed_id]),
                                     seed_cross=float(cross_sim[seed_id]),
                                     matched_clusters=int(matched_ids.numel()),
                                     selected_clusters=int((combined > self.merge_threshold).sum()))
        # Keep every cluster whose final score exceeds the merge threshold (0.2).
        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
        return final_mask
