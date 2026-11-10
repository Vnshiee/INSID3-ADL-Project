"""Task 2 — Prototype aggregation and region matching.

Notation
    x_i   (C,)   debiased, L2-normalised target token i ∈ {1..P}
    q_j   (C,)   reference prototype j ∈ {1..M}, weight w_j (Σ w_j = 1)
    s_ij = <x_i, q_j>                          token–prototype similarity  (P, M)
    G_k          target cluster k (agglomerative, original space), k ∈ {1..K}

Token-level aggregation  (used for the forward mask and the per-cluster cross score)
    max        a_i = max_j s_ij
    weighted   a_i = Σ_j w_j s_ij
    lse        a_i = T · log Σ_j w_j exp(s_ij / T)          (weighted ≤ lse ≤ max)
    ('single' — the original prototype — is handled by the model.)

    Note: with *unnormalised* cluster means, the weighted rule collapses to the original
    single prototype by linearity (Σ_j w_j mean_j = global mean). Normalising each q_j
    is therefore what makes 'weighted' differ from the baseline; 'max'/'lse' are
    genuinely non-linear.

Region-level (many-to-many) matching of a cluster G_k with the prototype set
    chamfer    r_k = ½ [ mean_{i∈G_k} max_j s_ij  +  Σ_j w_j max_{i∈G_k} s_ij ]
    ot         r_k = Σ_{i∈G_k, j} π^k_ij s_ij / Σ π^k_ij, π^k = unbalanced entropic OT plan
               between uniform mass on G_k (hard marginal) and w (KL-relaxed, weight ρ).
               ρ → 0  : π becomes a per-token softmax assignment (≈ lse with T = ε)
               ρ → ∞  : balanced OT — the cluster must "cover" every prototype.

All functions are batched over clusters with scatter operations (no Python loop over K).
"""

from __future__ import annotations

import torch


# ───────────────────────────── segment (per-cluster) reductions ─────────────────────────────

def segment_count(labels: torch.Tensor, K: int) -> torch.Tensor:
    """(K,) number of tokens per cluster."""
    return torch.bincount(labels, minlength=K)


def segment_mean(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Per-cluster mean. values (P,) or (P, M) → (K,) or (K, M); empty clusters → 0."""
    shape = (K,) + tuple(values.shape[1:])
    sums = torch.zeros(shape, device=values.device, dtype=values.dtype).index_add_(0, labels, values)
    cnt = segment_count(labels, K).to(values.dtype).clamp_min(1)
    return sums / cnt.view((K,) + (1,) * (values.dim() - 1))


def segment_amax(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Per-cluster max. values (P, M) → (K, M); empty clusters → -inf."""
    M = values.shape[1]
    out = torch.full((K, M), float("-inf"), device=values.device, dtype=values.dtype)
    return out.scatter_reduce(0, labels[:, None].expand(-1, M), values, reduce="amax", include_self=True)


def segment_logsumexp(values: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Per-cluster log Σ exp. values (P, M) → (K, M); empty clusters → -inf."""
    mx = segment_amax(values, labels, K)                              # (K, M)
    safe = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    ex = torch.exp(values - safe[labels])                             # (P, M)
    s = torch.zeros_like(mx).index_add_(0, labels, ex)
    return safe + torch.log(s)                                        # log 0 = -inf for empty


# ───────────────────────────── token-level aggregation ─────────────────────────────

def aggregate_similarities(sims: torch.Tensor, weights: torch.Tensor, mode: str,
                           temperature: float = 0.05) -> torch.Tensor:
    """Aggregate prototype similarities along the last dimension.

    Args:
        sims: (..., M) similarities to the M prototypes.
        weights: (M,) prototype masses, Σ = 1.
        mode: max | weighted | lse.
        temperature: T for lse.
    Returns:
        (...) aggregated similarity.
    """
    w = weights.to(device=sims.device, dtype=sims.dtype)
    if mode == "max":
        return sims.max(dim=-1).values
    if mode == "weighted":
        return (sims * w).sum(dim=-1)
    if mode == "lse":
        logw = torch.log(w.clamp_min(1e-12))
        return temperature * torch.logsumexp(sims / temperature + logw, dim=-1)
    raise ValueError(f"Unknown token aggregation: {mode}")


# ───────────────────────────── region-level many-to-many matching ─────────────────────────────

def region_scores_chamfer(sims: torch.Tensor, labels: torch.Tensor, K: int,
                          weights: torch.Tensor) -> torch.Tensor:
    """Bidirectional (Chamfer) set similarity between each cluster and the prototype set.

    Args:
        sims: (P, M) token–prototype similarities; labels: (P,) cluster ids in [0, K).
    Returns:
        (K,) scores; empty clusters get 0.
    """
    t2r = segment_mean(sims.max(dim=1).values, labels, K)             # (K,)
    best_tok = segment_amax(sims, labels, K)                          # (K, M)
    best_tok = torch.where(torch.isfinite(best_tok), best_tok, torch.zeros_like(best_tok))
    r2t = best_tok @ weights.to(sims)                                 # (K,)
    return 0.5 * (t2r + r2t)


def unbalanced_sinkhorn_plan(sims: torch.Tensor, labels: torch.Tensor, K: int,
                             weights: torch.Tensor, epsilon: float = 0.05, rho: float = 0.5,
                             iters: int = 50) -> torch.Tensor:
    """Per-cluster entropic OT plans, all clusters solved jointly (log domain).

    For cluster k: source mass a_i = 1/|G_k| on its tokens (hard constraint),
    target mass w_j on prototypes (KL penalty ρ), cost C_ij = 1 − s_ij.
        π_ij = a_i w_j exp((f_i + g_{k(i) j} − C_ij)/ε)
        f_i  = −ε log Σ_j w_j exp((g_{k(i)j} − C_ij)/ε)                  (row constraint)
        g_kj = −ρ/(ρ+ε) · ε log Σ_{i∈G_k} a_i exp((f_i − C_ij)/ε)          (relaxed column)

    Args:
        sims: (P, M); labels: (P,) in [0, K); weights: (M,).
    Returns:
        plan: (P, M) with Σ_j plan_ij = a_i, so Σ_{i∈G_k, j} plan_ij = 1 for non-empty k.
    """
    sims = sims.float()
    P, M = sims.shape
    cost = 1.0 - sims                                                  # (P, M)
    cnt = segment_count(labels, K).float().clamp_min(1)                # (K,)
    log_a = -torch.log(cnt)[labels]                                    # (P,)
    log_w = torch.log(weights.to(sims).clamp_min(1e-12))               # (M,)
    g = torch.zeros(K, M, device=sims.device)
    scale = rho / (rho + epsilon)

    def f_update(g_):
        return -epsilon * torch.logsumexp(log_w[None] + (g_[labels] - cost) / epsilon, dim=1)

    for _ in range(iters):
        f = f_update(g)                                                # (P,)
        lse = segment_logsumexp(log_a[:, None] + (f[:, None] - cost) / epsilon, labels, K)
        g = -scale * epsilon * torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    f = f_update(g)
    log_plan = log_a[:, None] + log_w[None] + (f[:, None] + g[labels] - cost) / epsilon
    return torch.exp(log_plan)


def region_scores_ot(sims: torch.Tensor, labels: torch.Tensor, K: int, weights: torch.Tensor,
                     epsilon: float = 0.05, rho: float = 0.5, iters: int = 50) -> torch.Tensor:
    """(K,) transport-weighted similarity Σ π s / Σ π for every cluster (0 for empty)."""
    plan = unbalanced_sinkhorn_plan(sims, labels, K, weights, epsilon, rho, iters)
    num = torch.zeros(K, device=sims.device).index_add_(0, labels, (plan * sims.float()).sum(1))
    den = torch.zeros(K, device=sims.device).index_add_(0, labels, plan.sum(1))
    return (num / den.clamp_min(1e-12)).to(sims.dtype)
