"""Task 2 — Multi-prototype reference representation.

Original INSID3 (models/insid3.py:predict_mask, lines "Reference prototype")
    for each shot m: fg_m = debiased reference tokens inside the downsampled mask, (C, N_m)
    p̃_r = normalize( mean_m mean_j fg_m[:, j] )                        (C,)

Proposed
    For each shot, partition its N_m foreground tokens into M_m groups and keep one
    L2-normalised *debiased* mean per group:
        q_{m,j} = normalize( mean_{i ∈ G_{m,j}} x̃_i ),    w_{m,j} = |G_{m,j}| / (N_m · S)
    so the weights sum to 1 over all prototypes and every shot gets equal total mass
    (exactly the shot weighting of the original average).

    Grouping space (``cluster_space``): by default tokens are *grouped* with the
    original features (intra-image structure, as INSID3 does for the target) and
    *represented* with the debiased features (cross-image matching space).

Robustness
    * M_eff = max(1, min(M, ⌊N / min_points_per_prototype⌋)) — tiny masks collapse to the
      original single prototype.
    * k-means++ initialisation with a *local* ``torch.Generator``: the global numpy / torch
      RNG is never touched (the official datasets draw episodes from the global numpy RNG,
      so touching it would change the evaluation episodes).
    * Empty clusters are re-seeded with the point worst explained by its centroid.
    * Clusters smaller than ``min_cluster_fraction · N`` are dropped (mass renormalised).
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
    """Reference prototypes.

    Attributes:
        prototypes: (M, C) L2-normalised prototypes in the matching (debiased) space.
        weights:    (M,)   non-negative, sum to 1.
        shot_index: (M,)   which reference image each prototype comes from.
        global_prototype: (C,) the original INSID3 prototype p̃_r (always computed).
    """
    prototypes: torch.Tensor
    weights: torch.Tensor
    shot_index: torch.Tensor
    global_prototype: torch.Tensor

    @property
    def num(self) -> int:
        return int(self.prototypes.shape[0])


def spherical_kmeans(X: torch.Tensor, k: int, iters: int = 50,
                     generator: Optional[torch.Generator] = None, tol: float = 1e-6):
    """Cosine k-means on the unit sphere.

    Args:
        X: (N, C) L2-normalised points.
        k: number of clusters, 1 ≤ k ≤ N.
        iters: maximum Lloyd iterations.
        generator: CPU generator for the k-means++ draws (deterministic).
    Returns:
        labels (N,) long, centers (k, C) L2-normalised, objective = mean cosine to centre.

    Empty clusters are re-seeded during the iterations. The returned labels are the final
    nearest-centre assignment, so with degenerate input (e.g. duplicated points) a cluster
    can still end up empty; callers must drop empty clusters (build_reference_prototypes does).
    """
    N = X.shape[0]
    if not 1 <= k <= N:
        raise ValueError(f"k={k} must be in [1, N={N}]")
    Xf = X.float()
    # k-means++ seeding with cosine distance d = 1 - cos.
    first = int(torch.randint(N, (1,), generator=generator).item())
    centers = [Xf[first]]
    dist = (1.0 - Xf @ Xf[first]).clamp_min(0)
    for _ in range(1, k):
        probs = dist.cpu().double()
        total = float(probs.sum())
        if total <= 0:                     # all remaining points coincide with a centre
            idx = int(torch.randint(N, (1,), generator=generator).item())
        else:
            idx = int(torch.multinomial(probs / total, 1, generator=generator).item())
        centers.append(Xf[idx])
        dist = torch.minimum(dist, (1.0 - Xf @ Xf[idx]).clamp_min(0))
    C = torch.stack(centers)                                         # (k, C)

    labels = torch.zeros(N, dtype=torch.long, device=X.device)
    prev = None
    for _ in range(iters):
        sim = Xf @ C.T                                               # (N, k)
        best, labels = sim.max(dim=1)
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros_like(C).index_add_(0, labels, Xf)
        for j in torch.nonzero(counts == 0).flatten().tolist():      # re-seed empty clusters
            worst = int(torch.argmin(best).item())
            sums[j] = Xf[worst]
            labels[worst] = j
            best[worst] = 1.0
        C = F.normalize(sums, p=2, dim=1)
        obj = float(best.mean())
        if prev is not None and abs(obj - prev) < tol:
            break
        prev = obj
    sim = Xf @ C.T
    best, labels = sim.max(dim=1)
    return labels, C, float(best.mean())


def _group_tokens(Xc: torch.Tensor, cfg, generator: torch.Generator) -> torch.Tensor:
    """Return (N,) group labels in [0, M_eff) for one shot (grouping-space features Xc)."""
    N = Xc.shape[0]
    m_eff = max(1, min(cfg.num_prototypes, N // cfg.min_points_per_prototype))
    if m_eff == 1:
        return torch.zeros(N, dtype=torch.long, device=Xc.device)

    if cfg.clustering == "kmeans":
        best = None
        for _ in range(cfg.kmeans_restarts):
            labels, _, obj = spherical_kmeans(Xc, m_eff, cfg.kmeans_iters, generator)
            if best is None or obj > best[1]:
                best = (labels, obj)
        return best[0]

    # Agglomerative (same routine/linkage as INSID3's target clustering), adaptive count,
    # capped at num_prototypes by keeping the largest groups and reassigning the rest.
    labels = agglomerative_clustering(F.normalize(Xc.float(), dim=1), cfg.ref_tau)
    counts = torch.bincount(labels)
    if counts.numel() > m_eff:
        keep = torch.argsort(counts, descending=True)[:m_eff]
        cents = F.normalize(torch.stack([Xc[labels == k].float().mean(0) for k in keep.tolist()]), dim=1)
        labels = (F.normalize(Xc.float(), dim=1) @ cents.T).argmax(1)
    else:
        _, labels = torch.unique(labels, return_inverse=True)
    return labels


def build_reference_prototypes(feat_match: torch.Tensor, feat_group: torch.Tensor,
                               ref_masks: torch.Tensor, cfg) -> PrototypeSet:
    """Build multi-prototype reference sets for all shots.

    Args:
        feat_match: (S, C, h, w) features of the matching space (debiased, L2-normalised).
        feat_group: (S, C, h, w) features used for grouping (original or debiased).
        ref_masks:  (S, 1, H, W) boolean reference masks at image resolution.
        cfg: :class:`insid3_ext.config.PrototypeConfig`.
    Returns:
        PrototypeSet with M ≥ S prototypes (≥ 1 per shot whose mask is non-empty).
    """
    S, C, h, w = feat_match.shape
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    protos: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    shots: List[torch.Tensor] = []
    shot_means: List[torch.Tensor] = []

    for s in range(S):
        mask_s = downsample_mask(ref_masks[s:s + 1], h, w)          # (h, w) bool, official routine
        fg_cn = feat_match[s, :, mask_s]                             # (C, N), as in the official code
        if fg_cn.shape[1] == 0:
            continue
        shot_means.append(fg_cn.mean(dim=1))                         # original per-shot mean (C,)
        fg = fg_cn.T                                                 # (N, C)
        Xg = feat_group[s, :, mask_s].T                              # (N, C)
        labels = _group_tokens(Xg, cfg, gen).to(fg.device)
        M = int(labels.max().item()) + 1
        counts = torch.bincount(labels, minlength=M).float()         # (M,)
        # fp32 accumulation (features may be bf16 under CUDA autocast)
        sums = torch.zeros(M, C, device=fg.device, dtype=torch.float32).index_add_(0, labels, fg.float())
        keep = counts > 0
        if cfg.min_cluster_fraction > 0 and int(keep.sum()) > 1:
            big = counts >= cfg.min_cluster_fraction * fg.shape[0]
            if bool(big.any()):
                keep = keep & big
        q = F.normalize(sums[keep], p=2, dim=1)                      # (M', C)
        wq = counts[keep] / counts[keep].sum()                       # (M',)
        protos.append(q)
        weights.append(wq)
        shots.append(torch.full((q.shape[0],), s, dtype=torch.long, device=fg.device))

    if not protos:
        raise ValueError("All reference masks are empty at feature resolution.")

    # Original single prototype (identical expression to the official code).
    global_proto = F.normalize(torch.stack(shot_means).mean(dim=0), p=2, dim=0)  # (C,)

    n_shots = len(protos)
    Q = torch.cat(protos)                                            # (M, C)
    W = torch.cat([wq / n_shots for wq in weights])                  # (M,) sums to 1
    I = torch.cat(shots)
    if cfg.include_global:
        Q = torch.cat([Q, global_proto[None].to(Q.dtype)])
        W = torch.cat([W * (1 - cfg.global_weight), W.new_tensor([cfg.global_weight])])
        I = torch.cat([I, I.new_tensor([-1])])
    return PrototypeSet(prototypes=Q, weights=W.float(), shot_index=I, global_prototype=global_proto)
