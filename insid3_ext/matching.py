"""Task 2 — scoring target patches and target regions against SEVERAL prototypes.

With one prototype p (original INSID3) the score of a target patch x is simply the cosine <x, p>.
With prototypes q_1..q_M (weights w_1..w_M) we first get M similarities per patch,

        s_ij = <x_i, q_j>          sims: (P, M) for P target patches

and must combine them into one score. Two levels:

A) PATCH LEVEL (used for INSID3's "forward" filter and for the cluster score in the final merging)
     max        a_i = max_j s_ij                 "matches its best part"
     weighted   a_i = sum_j w_j s_ij             "matches the average over parts"
     lse        a_i = T log sum_j w_j exp(s_ij / T)   a smooth max between the two
                (always weighted <= lse <= max; small T -> max, large T -> weighted)
     ('single' = the original prototype only; handled in model.py)

   Why prototypes are unit-length: if they were plain (unnormalised) group means, 'weighted' would be
   EXACTLY the original single prototype (a weighted sum of group means is the overall mean). So the
   only differences to the original come from normalising each group and from max / lse.

B) REGION LEVEL (target clusters G_k from INSID3's clustering; used for seed choice and merging)
     chamfer    r_k = 1/2 [ mean over patches of the best prototype     ("each patch fits some part")
                          + weighted mean over prototypes of the best patch ]  ("each part is found")
     ot         optimal transport: the cluster's patches send their mass to the prototypes; the score
                is the average similarity along the transport. See unbalanced_sinkhorn_plan.

All functions handle all K clusters at once with "segment" operations (sum / max over the patches
that share a cluster label) instead of a Python loop over clusters.
"""

from __future__ import annotations

import torch


# ════════════════════════════════════════════════════════════════════════════════════════
# 1. Per-cluster reductions (labels (P,) say which cluster each patch belongs to)
# ════════════════════════════════════════════════════════════════════════════════════════

def segment_count(labels: torch.Tensor, K: int) -> torch.Tensor:
    """Number of patches in each of the K clusters, (K,)."""
    return torch.bincount(labels, minlength=K)


def segment_mean(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Average of `values` over each cluster. values (P,) or (P, M) -> (K,) or (K, M); empty -> 0."""
    shape = (K,) + tuple(values.shape[1:])
    sums = torch.zeros(shape, device=values.device, dtype=values.dtype).index_add_(0, labels, values)
    cnt = segment_count(labels, K).to(values.dtype).clamp_min(1)     # clamp: avoid 0/0 for empty clusters
    return sums / cnt.view((K,) + (1,) * (values.dim() - 1))


def segment_amax(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Maximum of `values` (P, M) over each cluster -> (K, M); empty clusters -> -inf."""
    M = values.shape[1]
    out = torch.full((K, M), float("-inf"), device=values.device, dtype=values.dtype)
    return out.scatter_reduce(0, labels[:, None].expand(-1, M), values, reduce="amax", include_self=True)


def segment_logsumexp(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """log(sum(exp(values))) over each cluster, (P, M) -> (K, M); empty -> -inf.

    Computed stably: subtract the cluster maximum before exp so nothing overflows.
    """
    mx = segment_amax(values, labels, K)                              # (K, M)
    safe = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    ex = torch.exp(values - safe[labels])                             # (P, M)
    s = torch.zeros_like(mx).index_add_(0, labels, ex)
    return safe + torch.log(s)


# ════════════════════════════════════════════════════════════════════════════════════════
# 2. Patch level: combine the M similarities of each patch into one score
# ════════════════════════════════════════════════════════════════════════════════════════

def aggregate_similarities(sims: torch.Tensor, weights: torch.Tensor, mode: str,
                           temperature: float = 0.05) -> torch.Tensor:
    """sims (..., M), weights (M,) summing to 1 -> one score per row (...)."""
    w = weights.to(device=sims.device, dtype=sims.dtype)
    if mode == "max":
        return sims.max(dim=-1).values
    if mode == "weighted":
        return (sims * w).sum(dim=-1)
    if mode == "lse":
        # T * log( sum_j w_j * exp(s_j / T) ), written with log w_j so logsumexp keeps it stable.
        logw = torch.log(w.clamp_min(1e-12))
        return temperature * torch.logsumexp(sims / temperature + logw, dim=-1)
    raise ValueError(f"Unknown token aggregation: {mode}")


# ════════════════════════════════════════════════════════════════════════════════════════
# 3. Region level: score a whole target cluster against the whole prototype set
# ════════════════════════════════════════════════════════════════════════════════════════

def region_scores_chamfer(sims: torch.Tensor, labels: torch.Tensor, K: int,
                          weights: torch.Tensor) -> torch.Tensor:
    """Chamfer score per cluster, (K,). sims (P, M), labels (P,)."""
    # Direction 1: for every patch its best prototype, averaged over the cluster's patches.
    t2r = segment_mean(sims.max(dim=1).values, labels, K)             # (K,)
    # Direction 2: for every prototype its best patch in the cluster, weighted over prototypes.
    best_tok = segment_amax(sims, labels, K)                          # (K, M)
    best_tok = torch.where(torch.isfinite(best_tok), best_tok, torch.zeros_like(best_tok))
    r2t = best_tok @ weights.to(sims)                                 # (K,)
    return 0.5 * (t2r + r2t)


def unbalanced_sinkhorn_plan(sims: torch.Tensor, labels: torch.Tensor, K: int,
                             weights: torch.Tensor, epsilon: float = 0.05, rho: float = 0.5,
                             iters: int = 50) -> torch.Tensor:
    """Transport plan between each cluster's patches and the prototypes, (P, M).

    IDEA. For one cluster: every patch holds mass 1/|G_k| and must send ALL of it to the prototypes
    (hard rule). Prototype j "wants" mass w_j, but this is only a soft wish: missing or extra mass
    costs rho * KL (a penalty). Sending mass from patch i to prototype j costs 1 - s_ij. epsilon adds
    entropy (spreads the mass a bit), which makes the problem solvable with the Sinkhorn iterations.

      rho -> 0   : the prototypes' wishes are ignored -> each patch sends its mass to its best
                   prototypes (a soft max per patch, like lse with T = epsilon)
      rho -> inf : the wishes become hard -> the cluster must cover EVERY prototype in proportion w
                   (classical balanced optimal transport)

    COMPUTATION (log domain = numbers stay finite). With potentials f (one per patch) and
    g (one per cluster and prototype) the plan is

        plan_ij = a_i * w_j * exp((f_i + g_kj - C_ij) / epsilon),   C_ij = 1 - s_ij,  a_i = 1/|G_k|

    and the updates alternate (Peyre & Cuturi 2019, eq. 10.9, here in log form):
        f_i  = -epsilon log sum_j w_j exp((g_kj - C_ij)/epsilon)                    patches: hard rule
        g_kj = -rho/(rho+epsilon) * epsilon log sum_{i in G_k} a_i exp((f_i - C_ij)/epsilon)   soft rule
    The factor rho/(rho+epsilon) is exactly what turns the hard rule into the soft KL rule.
    All K clusters are solved together; each patch only interacts with its own cluster's g.
    """
    sims = sims.float()
    P, M = sims.shape
    cost = 1.0 - sims                                                  # (P, M)
    cnt = segment_count(labels, K).float().clamp_min(1)                # (K,)
    log_a = -torch.log(cnt)[labels]                                    # (P,) log(1/|G_k|) per patch
    log_w = torch.log(weights.to(sims).clamp_min(1e-12))               # (M,)
    g = torch.zeros(K, M, device=sims.device)
    scale = rho / (rho + epsilon)

    def f_update(g_):
        return -epsilon * torch.logsumexp(log_w[None] + (g_[labels] - cost) / epsilon, dim=1)

    for _ in range(iters):
        f = f_update(g)                                                # (P,)
        lse = segment_logsumexp(log_a[:, None] + (f[:, None] - cost) / epsilon, labels, K)
        g = -scale * epsilon * torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    f = f_update(g)                                                    # last f: the patch rule holds exactly
    log_plan = log_a[:, None] + log_w[None] + (f[:, None] + g[labels] - cost) / epsilon
    return torch.exp(log_plan)


def region_scores_ot(sims: torch.Tensor, labels: torch.Tensor, K: int, weights: torch.Tensor,
                     epsilon: float = 0.05, rho: float = 0.5, iters: int = 50) -> torch.Tensor:
    """OT score per cluster (K,): average similarity along the plan, sum(plan * s) / sum(plan)."""
    plan = unbalanced_sinkhorn_plan(sims, labels, K, weights, epsilon, rho, iters)
    num = torch.zeros(K, device=sims.device).index_add_(0, labels, (plan * sims.float()).sum(1))
    den = torch.zeros(K, device=sims.device).index_add_(0, labels, plan.sum(1))
    return (num / den.clamp_min(1e-12)).to(sims.dtype)
