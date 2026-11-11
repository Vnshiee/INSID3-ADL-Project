"""Task 2 — scoring target patches and target regions against SEVERAL prototypes.

With one prototype p (original INSID3) the score of a target patch x is simply the cosine <x, p>.
With prototypes q_1..q_M (weights w_1..w_M) every patch gets M similarities

        s_ij = <x_i, q_j>          sims: (P, M) for P target patches

and we must turn them into one score. Two levels:

A) PATCH LEVEL (used for INSID3's "forward" filter and for the cross score of every cluster)
     max        a_i = max_j s_ij                        "the patch matches its best part"
     weighted   a_i = sum_j w_j s_ij                    "the patch matches the average over parts"
     lse        a_i = T log sum_j w_j exp(s_ij / T)     a smooth max between the two
                (always weighted <= lse <= max; small T -> max, large T -> weighted)
     ('single' = the original prototype only; handled in model.py)

   Why the prototypes have unit length: if they were plain (unnormalised) group means, 'weighted'
   would be EXACTLY the original single prototype (a weighted sum of group means is the overall mean).
   So the only differences to the original come from normalising each group and from max / lse.

B) REGION LEVEL (target clusters G_1..G_K from INSID3's clustering; used for the seed and the merging)
     chamfer    r_k = 1/2 [ mean over the cluster's patches of their best prototype       ("every patch fits some part")
                          + weighted mean over prototypes of their best patch ]          ("every part is found")
     ot         optimal transport: the cluster's patches send their mass to the prototypes; the score is
                the average similarity along the transport. See sinkhorn_plan_one_region.

Every region function is a plain loop over the K clusters, like the official code (models/insid3.py
L352-354 computes its cross score with exactly such a loop).
"""

from __future__ import annotations

import torch


# ════════════════════════════════════════════════════════════════════════════════════════
# 1. Patch level: combine the M similarities of each patch into one score
# ════════════════════════════════════════════════════════════════════════════════════════

def aggregate_similarities(sims: torch.Tensor, weights: torch.Tensor, mode: str,
                           temperature: float = 0.05) -> torch.Tensor:
    """sims (..., M): similarities to the M prototypes; weights (M,) sum to 1 -> one score per row (...)."""
    w = weights.to(device=sims.device, dtype=sims.dtype)
    if mode == "max":
        return sims.max(dim=-1).values
    if mode == "weighted":
        return (sims * w).sum(dim=-1)
    if mode == "lse":
        # T * log( sum_j w_j * exp(s_j / T) ). We write w_j * exp(s_j / T) as exp(s_j / T + log w_j)
        # so that torch.logsumexp can compute it without overflow. clamp: log(0) would be -inf.
        log_w = torch.log(w.clamp_min(1e-12))
        return temperature * torch.logsumexp(sims / temperature + log_w, dim=-1)
    raise ValueError(f"Unknown token aggregation: {mode}")


def mean_per_cluster(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Average of `values` (P,) over the patches of each cluster -> (K,). An empty cluster gets 0.

    The same loop as the official cross score (models/insid3.py L351-354).
    """
    out = torch.zeros(K, device=values.device, dtype=values.dtype)
    for k in range(K):
        in_cluster = labels == k
        if in_cluster.any():
            out[k] = values[in_cluster].mean()
    return out


# ════════════════════════════════════════════════════════════════════════════════════════
# 2. Region level, Chamfer: each patch finds its best part, each part finds its best patch
# ════════════════════════════════════════════════════════════════════════════════════════

def region_scores_chamfer(sims: torch.Tensor, labels: torch.Tensor, K: int,
                          weights: torch.Tensor) -> torch.Tensor:
    """Chamfer score of every cluster, (K,). sims (P, M), labels (P,) cluster of every patch."""
    w = weights.to(sims)
    scores = torch.zeros(K, device=sims.device, dtype=sims.dtype)
    for k in range(K):
        s = sims[labels == k]                                   # (n, M): this cluster's patches only
        if s.shape[0] == 0:
            continue                                            # empty cluster: score stays 0
        patch_to_part = s.max(dim=1).values.mean()              # every patch: its best prototype, averaged
        part_to_patch = (s.max(dim=0).values * w).sum()         # every prototype: its best patch, weighted
        scores[k] = 0.5 * (patch_to_part + part_to_patch)
    return scores


# ════════════════════════════════════════════════════════════════════════════════════════
# 3. Region level, optimal transport (many-to-many matching)
# ════════════════════════════════════════════════════════════════════════════════════════

def sinkhorn_plan_one_region(s: torch.Tensor, weights: torch.Tensor, epsilon: float = 0.05,
                             rho: float = 0.5, iters: int = 50) -> torch.Tensor:
    """Transport plan between the n patches of ONE cluster and the M prototypes, (n, M).

    IDEA. Every patch holds mass 1/n and must send ALL of it to the prototypes (hard rule).
    Prototype j "wants" mass w_j, but only as a soft wish: missing or extra mass costs rho * KL.
    Sending mass from patch i to prototype j costs C_ij = 1 - s_ij. epsilon adds entropy (spreads the
    mass a little), which makes the problem solvable with simple alternating updates (Sinkhorn).

      rho -> 0   : the wishes are ignored -> each patch sends its mass to its best prototypes
                   (a soft max per patch, like lse with T = epsilon)
      rho -> inf : the wishes become hard -> the cluster must cover EVERY prototype in proportion w
                   (classical balanced optimal transport)

    COMPUTATION in the log domain (exp of large numbers would overflow). With one potential f_i per
    patch and g_j per prototype, the plan is   plan_ij = a_i * w_j * exp((f_i + g_j - C_ij) / epsilon)
    and the two updates alternate (Peyre & Cuturi 2019, eq. 10.9, written with logs):
        f_i = -epsilon * log sum_j w_j exp((g_j - C_ij) / epsilon)                  rows: hard rule
        g_j = -rho/(rho+epsilon) * epsilon * log sum_i a_i exp((f_i - C_ij) / epsilon)   columns: soft rule
    The factor rho/(rho+epsilon) is exactly what turns the hard column rule into the soft KL rule.
    """
    s = s.float()
    n, M = s.shape
    cost = 1.0 - s                                               # (n, M)
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n))).item(), device=s.device)   # log(1/n)
    log_w = torch.log(weights.to(s).clamp_min(1e-12))            # (M,)
    scale = rho / (rho + epsilon)

    g = torch.zeros(M, device=s.device)
    for _ in range(iters):
        # rows: every patch sends exactly its mass 1/n
        f = -epsilon * torch.logsumexp(log_w[None, :] + (g[None, :] - cost) / epsilon, dim=1)       # (n,)
        # columns: prototype j receives roughly w_j (soft rule)
        g = -scale * epsilon * torch.logsumexp(log_a[:, None] + (f[:, None] - cost) / epsilon, dim=0)  # (M,)
    # one last row update, so that the hard rule (every patch sends 1/n) holds exactly
    f = -epsilon * torch.logsumexp(log_w[None, :] + (g[None, :] - cost) / epsilon, dim=1)
    log_plan = log_a[:, None] + log_w[None, :] + (f[:, None] + g[None, :] - cost) / epsilon
    return torch.exp(log_plan)


def unbalanced_sinkhorn_plan(sims: torch.Tensor, labels: torch.Tensor, K: int, weights: torch.Tensor,
                             epsilon: float = 0.05, rho: float = 0.5, iters: int = 50) -> torch.Tensor:
    """Transport plans of all clusters, written into one (P, M) tensor (row i = patch i).

    Not needed for the score itself; the tests use it to check the plan (row sums, limits of rho).
    """
    plan = torch.zeros(sims.shape, device=sims.device, dtype=torch.float32)
    for k in range(K):
        in_cluster = labels == k
        if in_cluster.any():
            plan[in_cluster] = sinkhorn_plan_one_region(sims[in_cluster], weights, epsilon, rho, iters)
    return plan


def region_scores_ot(sims: torch.Tensor, labels: torch.Tensor, K: int, weights: torch.Tensor,
                     epsilon: float = 0.05, rho: float = 0.5, iters: int = 50) -> torch.Tensor:
    """OT score of every cluster, (K,): the average similarity along its plan, sum(plan * s) / sum(plan)."""
    scores = torch.zeros(K, device=sims.device)
    for k in range(K):
        s = sims[labels == k].float()
        if s.shape[0] == 0:
            continue                                            # empty cluster: score stays 0
        plan = sinkhorn_plan_one_region(s, weights, epsilon, rho, iters)
        scores[k] = (plan * s).sum() / plan.sum().clamp_min(1e-12)
    return scores.to(sims.dtype)
