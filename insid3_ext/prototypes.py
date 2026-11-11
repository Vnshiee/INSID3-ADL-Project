"""Task 2 — describing the reference region with several prototypes instead of one.

WHAT THE ORIGINAL INSID3 DOES (models/insid3.py lines 150-158)
For each reference image (shot) take the debiased features of the patches inside its mask and
average them; average those per-shot means; scale to unit length. That single vector p is the
"prototype" of the object.

WHY WE CHANGE IT
One average can sit far away from every part of a multi-part object (head, body, legs ...).
So we split the masked patches into groups of similar patches and keep one average per group.

HOW
  1. Patches inside the mask -> M groups (spherical k-means, or agglomerative clustering).
     Grouping uses the ORIGINAL features by default: INSID3 itself groups the target image with
     original features (they keep spatial structure); only cross-image MATCHING needs debiasing.
  2. Each group -> one prototype q_j = unit-length mean of its DEBIASED features (the space in
     which reference and target are compared).
  3. Each prototype gets a weight w_j = its share of the patches. Every shot gets the same total
     weight 1/S, just as the original averages shots with equal weight. Weights sum to 1.

Two ways to handle several shots (prototypes.multishot):
  per_shot  cluster each shot separately (S x M prototypes)                    [round-1 behaviour]
  pooled    cluster the patches of all shots together, each patch weighted 1/(S * N_s)
            (ASGNet does this). With M = 1 this gives exactly the original prototype for any S.

Safety rules
  * M is reduced for small masks: M_eff = min(M, N // min_points_per_prototype), at least 1, so a
    tiny mask falls back to the single original prototype.
  * k-means uses its OWN random generator. The official datasets draw their episodes from numpy's
    global random generator; touching a global generator here could change the test episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from utils.clustering import agglomerative_clustering
from utils.data import downsample_mask


@dataclass
class PrototypeSet:
    """The reference description used for matching.

    prototypes        (M, C) unit-length prototypes (debiased space)
    weights           (M,)   their weights, summing to 1
    shot_index        (M,)   which shot each one came from (-1 = global prototype, -2 = pooled)
    global_prototype  (C,)   the ORIGINAL single INSID3 prototype, always computed alongside
    """
    prototypes: torch.Tensor
    weights: torch.Tensor
    shot_index: torch.Tensor
    global_prototype: torch.Tensor

    @property
    def num(self) -> int:
        return int(self.prototypes.shape[0])


# ════════════════════════════════════════════════════════════════════════════════════════
# 1. Spherical k-means (k-means for unit vectors, distance = 1 - cosine)
# ════════════════════════════════════════════════════════════════════════════════════════

def spherical_kmeans(X: torch.Tensor, k: int, iters: int = 50,
                     generator: Optional[torch.Generator] = None, tol: float = 1e-6,
                     weights: Optional[torch.Tensor] = None):
    """Split N unit vectors X (N, C) into k groups of similar direction.

    Same as ordinary k-means, but "close" means "high cosine similarity" (the similarity INSID3
    uses everywhere) and every centre is rescaled to unit length.
    weights (N,) lets some points count more (used by the pooled multi-shot variant).

    Returns labels (N,), centres (k, C), objective = weighted mean cosine to the own centre.
    Note: with degenerate input (identical points) a group can end up empty; the caller drops it.
    """
    N = X.shape[0]
    if not 1 <= k <= N:
        raise ValueError(f"k={k} must be in [1, N={N}]")
    Xf = X.float()
    wt = torch.ones(N, device=X.device) if weights is None else weights.float().to(X.device)

    # --- k-means++ start: first centre random, every further centre preferably FAR from the
    #     existing ones (probability ~ distance x weight). Gives better and more stable groups
    #     than a purely random start.
    first = int(torch.randint(N, (1,), generator=generator).item())
    centers = [Xf[first]]
    dist = (1.0 - Xf @ Xf[first]).clamp_min(0)                   # distance to the nearest centre so far
    for _ in range(1, k):
        probs = (dist * wt).cpu().double()
        total = float(probs.sum())
        if total <= 0:                                           # every point already sits on a centre
            idx = int(torch.randint(N, (1,), generator=generator).item())
        else:
            idx = int(torch.multinomial(probs / total, 1, generator=generator).item())
        centers.append(Xf[idx])
        dist = torch.minimum(dist, (1.0 - Xf @ Xf[idx]).clamp_min(0))
    C = torch.stack(centers)                                     # (k, C)

    # --- Lloyd iterations: assign each point to its most similar centre, recompute centres.
    prev = None
    for _ in range(iters):
        sim = Xf @ C.T                                           # (N, k) cosine to every centre
        best, labels = sim.max(dim=1)                            # best centre of every point
        assigned = labels.clone()                                # groups as assigned in THIS iteration
        new_centers = []
        for j in range(k):
            members = assigned == j
            if members.any():
                # weighted sum of the group's points (normalised below = spherical k-means centre)
                new_centers.append((Xf[members] * wt[members, None]).sum(dim=0))
            else:
                # An empty group would have no centre: give it the point that fits its own centre worst.
                worst = int(torch.argmin(best).item())
                new_centers.append(Xf[worst])
                labels[worst] = j
                best[worst] = 1.0
        C = F.normalize(torch.stack(new_centers), p=2, dim=1)    # new centres, unit length
        obj = float((best * wt).sum() / wt.sum())
        if prev is not None and abs(obj - prev) < tol:           # stop when nothing improves any more
            break
        prev = obj
    sim = Xf @ C.T
    best, labels = sim.max(dim=1)
    return labels, C, float((best * wt).sum() / wt.sum())


# ════════════════════════════════════════════════════════════════════════════════════════
# 2. Grouping the patches of the reference mask
# ════════════════════════════════════════════════════════════════════════════════════════

def _group_tokens(Xc: torch.Tensor, cfg, generator: torch.Generator,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Group labels (N,) in [0, M_eff) for the N masked patches Xc (N, C)."""
    N = Xc.shape[0]
    # Small masks get fewer groups: at least min_points_per_prototype patches per group.
    m_eff = max(1, min(cfg.num_prototypes, N // cfg.min_points_per_prototype))
    if m_eff == 1:
        return torch.zeros(N, dtype=torch.long, device=Xc.device)   # one group = the original prototype

    if cfg.clustering == "kmeans":
        # Optionally several k-means runs with different starts; keep the best-fitting one.
        best = None
        for _ in range(cfg.kmeans_restarts):
            labels, _, obj = spherical_kmeans(Xc, m_eff, cfg.kmeans_iters, generator, weights=weights)
            if best is None or obj > best[1]:
                best = (labels, obj)
        return best[0]

    # Agglomerative: the same routine INSID3 uses for the target image, with threshold ref_tau.
    # The number of groups comes out of the data; if it exceeds m_eff we keep the m_eff largest
    # groups and give every other patch to the most similar kept group.
    labels = agglomerative_clustering(F.normalize(Xc.float(), dim=1), cfg.ref_tau)
    counts = torch.bincount(labels)                              # counts[j] = number of patches in group j
    if counts.numel() > m_eff:
        keep = torch.argsort(counts, descending=True)[:m_eff]
        cents = F.normalize(torch.stack([Xc[labels == k].float().mean(0) for k in keep.tolist()]), dim=1)
        labels = (F.normalize(Xc.float(), dim=1) @ cents.T).argmax(1)
    else:
        _, labels = torch.unique(labels, return_inverse=True)       # renumber 0..M-1
    return labels


def _masked_tokens(feat_match, feat_group, ref_masks, s, h, w):
    """Patches of shot s inside its mask: (C, N) matching features and (N, C) grouping features.

    The mask is shrunk to the patch grid with the official downsample_mask, as in INSID3.
    """
    mask_s = downsample_mask(ref_masks[s:s + 1], h, w)              # (h, w) bool
    return feat_match[s, :, mask_s], feat_group[s, :, mask_s].T


def _add_global(Q, W, I, global_proto, cfg):
    """Optionally keep the original prototype as one extra prototype with weight global_weight."""
    if cfg.include_global:
        Q = torch.cat([Q, global_proto[None].to(Q.dtype)])
        W = torch.cat([W * (1 - cfg.global_weight), W.new_tensor([cfg.global_weight])])
        I = torch.cat([I, I.new_tensor([-1])])
    return Q, W, I


# ════════════════════════════════════════════════════════════════════════════════════════
# 3. Building the prototype set
# ════════════════════════════════════════════════════════════════════════════════════════

def build_reference_prototypes(feat_match: torch.Tensor, feat_group: torch.Tensor,
                               ref_masks: torch.Tensor, cfg) -> PrototypeSet:
    """Prototypes for all shots.

    feat_match  (S, C, h, w)  debiased unit features: the prototypes are made of these
    feat_group  (S, C, h, w)  features used only to decide the groups (original by default)
    ref_masks   (S, 1, H, W)  reference masks at image resolution
    cfg         PrototypeConfig
    """
    S, C, h, w = feat_match.shape
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))  # own generator (see top of file)
    if cfg.multishot == "pooled" and S > 1:
        return _build_pooled(feat_match, feat_group, ref_masks, cfg, gen)

    protos: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    shots: List[torch.Tensor] = []
    shot_means: List[torch.Tensor] = []
    for s in range(S):
        fg_cn, Xg = _masked_tokens(feat_match, feat_group, ref_masks, s, h, w)   # (C, N), (N, C)
        if fg_cn.shape[1] == 0:
            continue
        shot_means.append(fg_cn.mean(dim=1))       # this shot's plain average (for the original prototype)
        fg = fg_cn.T                                # (N, C)

        labels = _group_tokens(Xg, cfg, gen).to(fg.device)
        M = int(labels.max().item()) + 1
        # Per group: number of patches and the sum of their debiased features (fp32 for accuracy).
        counts = torch.zeros(M, device=fg.device)
        sums = torch.zeros(M, C, device=fg.device, dtype=torch.float32)
        for j in range(M):
            members = labels == j
            counts[j] = members.sum()
            sums[j] = fg[members].float().sum(dim=0)

        # Drop empty groups, and (optionally) groups smaller than min_cluster_fraction of the mask.
        keep = counts > 0
        if cfg.min_cluster_fraction > 0 and int(keep.sum()) > 1:
            big = counts >= cfg.min_cluster_fraction * fg.shape[0]
            if bool(big.any()):
                keep = keep & big
        protos.append(F.normalize(sums[keep], p=2, dim=1))           # unit-length group means
        weights.append(counts[keep] / counts[keep].sum())             # share of this shot's patches
        shots.append(torch.full((int(keep.sum()),), s, dtype=torch.long, device=fg.device))

    if not protos:
        raise ValueError("All reference masks are empty at feature resolution.")

    # The original INSID3 prototype: unit-length average of the per-shot averages (same expression
    # as models/insid3.py line 156).
    global_proto = F.normalize(torch.stack(shot_means).mean(dim=0), p=2, dim=0)    # (C,)

    n_shots = len(protos)
    Q = torch.cat(protos)                                             # (M_total, C)
    W = torch.cat([wq / n_shots for wq in weights])                   # every shot weighs 1/S in total
    I = torch.cat(shots)
    Q, W, I = _add_global(Q, W, I, global_proto, cfg)
    return PrototypeSet(prototypes=Q, weights=W.float(), shot_index=I, global_prototype=global_proto)


def _build_pooled(feat_match, feat_group, ref_masks, cfg, gen) -> PrototypeSet:
    """Pooled multi-shot variant: one clustering over the patches of all shots.

    Patch i of shot s gets weight 1/(S' * N_s) (S' = shots with a non-empty mask), so every shot
    counts equally however large its mask is. Prototypes are weighted means; with M = 1 this is
    exactly the original prototype, average of the per-shot averages.
    """
    S, C, h, w = feat_match.shape
    xm, xg, wt, means = [], [], [], []
    for s in range(S):
        fg_cn, Xg = _masked_tokens(feat_match, feat_group, ref_masks, s, h, w)
        if fg_cn.shape[1] == 0:
            continue
        means.append(fg_cn.mean(dim=1))
        xm.append(fg_cn.T)
        xg.append(Xg)
        wt.append(torch.full((fg_cn.shape[1],), 1.0 / fg_cn.shape[1], device=fg_cn.device))
    if not xm:
        raise ValueError("All reference masks are empty at feature resolution.")

    n_shots = len(xm)
    Xm, Xg = torch.cat(xm), torch.cat(xg)                             # all masked patches of all shots
    Wt = torch.cat(wt) / n_shots                                      # patch weights, sum to 1
    global_proto = F.normalize(torch.stack(means).mean(dim=0), p=2, dim=0)

    labels = _group_tokens(Xg, cfg, gen, weights=Wt).to(Xm.device)
    M = int(labels.max().item()) + 1
    # Per group: its total patch weight ("mass") and the weighted sum of its features.
    mass = torch.zeros(M, device=Xm.device)
    sums = torch.zeros(M, C, device=Xm.device)
    for j in range(M):
        members = labels == j
        mass[j] = Wt[members].sum()
        sums[j] = (Xm[members].float() * Wt[members, None]).sum(dim=0)

    keep = mass > 0
    if cfg.min_cluster_fraction > 0 and int(keep.sum()) > 1:
        big = mass >= cfg.min_cluster_fraction                       # mass is already a fraction
        if bool(big.any()):
            keep = keep & big
    Q = F.normalize(sums[keep], p=2, dim=1)
    W = mass[keep] / mass[keep].sum()
    I = torch.full((Q.shape[0],), -2, dtype=torch.long, device=Q.device)
    Q, W, I = _add_global(Q, W, I, global_proto, cfg)
    return PrototypeSet(prototypes=Q, weights=W.float(), shot_index=I, global_prototype=global_proto)
