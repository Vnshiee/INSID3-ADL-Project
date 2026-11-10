"""Task 1 — measuring the contribution of positional components.

For a set of natural calibration images I = 1..N (resized exactly like at inference)
and every token position p = 1..P, project the L2-normalised feature onto the
constant-image basis U (C × R):

        c_k(I, p) = u_kᵀ x(I, p).

Additive model:  c_k(I, p) = π_k(p) + e_k(I, p), where π_k(p) is a deterministic
function of position (the positional signal, zero mean over p) and e_k is image
content. We estimate, per component k,

    template        π_k(p)  = mean_I c_k(I,p) − mean_{I,p} c_k        (R, P)
    positional var  v_k^pos = mean_p π_k(p)²  − (bias correction)      (R,)
    second moment   m_k     = mean_{I,p} c_k(I,p)²                       (R,)
    positional share ρ_k    = v_k^pos / m_k ∈ [0, 1]                     (R,)

ρ_k is the MMSE-optimal attenuation of c_k when the positional part is to be
removed by a gain without intercept (docs/MATHEMATICAL_FORMULATION.md §2.4).

The bias correction: mean_I c_k(I,p) has sampling variance σ²_within(k,p)/N, which
inflates the naive between-position variance; we subtract its average.

Also stored for analysis (Task-1 instruction "measure their contribution across
different images and feature tokens"):
    per_image_second_moment   (N, R)  m_k(I) per image
    per_image_noise_corr      (N, R)  corr_p(c_k(I,·), constant-image template_k)
    token_energy_mean / _std  (P,)    ||U_sᵀ x_p||² — energy in the official s-dim subspace
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class PositionalStatsAccumulator:
    """Streaming accumulator of positional statistics (float64 sums)."""

    def __init__(self, U: torch.Tensor, noise_template: torch.Tensor, svd_components: int,
                 grid: tuple):
        """
        Args:
            U: (C, R) orthonormal basis (the constant-image SVD).
            noise_template: (R, P) coefficients of the centred constant image.
            svd_components: s; used for the per-token energy map.
            grid: (h, w) token grid.
        """
        self.U = U.float()
        self.noise_template = noise_template.float()
        self.s = int(svd_components)
        self.grid = tuple(grid)
        R, P = noise_template.shape
        dev = U.device
        self.sum_c = torch.zeros(R, P, dtype=torch.float64, device=dev)
        self.sum_c2 = torch.zeros(R, P, dtype=torch.float64, device=dev)
        self.sum_e = torch.zeros(P, dtype=torch.float64, device=dev)
        self.sum_e2 = torch.zeros(P, dtype=torch.float64, device=dev)
        self.n = 0
        self.per_image_m: List[torch.Tensor] = []
        self.per_image_corr: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        """Add a batch of L2-normalised feature maps.

        Args:
            feats: (N, C, h, w), L2-normalised over C, on the same grid as the basis.
        """
        N, C, h, w = feats.shape
        if (h, w) != self.grid:
            raise ValueError(f"Feature grid {(h, w)} != basis grid {self.grid}")
        X = feats.reshape(N, C, h * w).float().to(self.U.device)
        coeff = torch.einsum('cr,ncp->nrp', self.U, X)               # (N, R, P)
        self.sum_c += coeff.sum(0).double()
        self.sum_c2 += coeff.pow(2).sum(0).double()

        e = coeff[:, : self.s].pow(2).sum(1)                         # (N, P) energy in top-s subspace
        self.sum_e += e.sum(0).double()
        self.sum_e2 += e.pow(2).sum(0).double()

        self.per_image_m.append(coeff.pow(2).mean(2).half().cpu())   # (N, R)
        cc = coeff - coeff.mean(2, keepdim=True)
        tt = self.noise_template - self.noise_template.mean(1, keepdim=True)  # (R, P)
        corr = (cc * tt[None]).sum(2) / (cc.norm(dim=2) * tt.norm(dim=1)[None]).clamp_min(1e-12)
        self.per_image_corr.append(corr.half().cpu())               # (N, R)
        self.n += N

    def finalize(self, meta: Optional[Dict[str, Any]] = None, S: Optional[torch.Tensor] = None
                 ) -> Dict[str, Any]:
        """Return the calibration dictionary (all tensors on CPU)."""
        if self.n < 2:
            raise ValueError("Need at least 2 calibration images.")
        n = float(self.n)
        mean = self.sum_c / n                                        # (R, P) E_I[c | p]
        second = self.sum_c2 / n                                     # (R, P) E_I[c² | p]
        grand = mean.mean(1)                                         # (R,)
        template = mean - grand[:, None]                             # (R, P)
        within_unbiased = (second - mean.pow(2)).clamp_min(0) * n / (n - 1)
        between_naive = template.pow(2).mean(1)                      # (R,)
        pos_var = (between_naive - within_unbiased.mean(1) / n).clamp_min(0)
        second_moment = second.mean(1)                               # (R,) E_{I,p}[c²]
        total_var = second_moment - grand.pow(2)
        rho = (pos_var / second_moment.clamp_min(1e-12)).clamp(0, 1)
        e_mean = self.sum_e / n
        e_std = (self.sum_e2 / n - e_mean.pow(2)).clamp_min(0).sqrt()

        out = {
            "meta": dict(meta or {}, num_images=int(self.n), grid=list(self.grid),
                         svd_components=self.s),
            "U": self.U.cpu(),
            "S": None if S is None else S.float().cpu(),
            "rho": rho.float().cpu(),
            "pos_second_moment": pos_var.float().cpu(),
            "second_moment": second_moment.float().cpu(),
            "total_var": total_var.float().cpu(),
            "mean_coeff": grand.float().cpu(),
            "template": template.float().cpu(),
            "token_energy_mean": e_mean.float().cpu(),
            "token_energy_std": e_std.float().cpu(),
            "per_image_second_moment": torch.cat(self.per_image_m),
            "per_image_noise_corr": torch.cat(self.per_image_corr),
        }
        return out


def summarize_calibration(cal: Dict[str, Any], svd_components: int) -> Dict[str, float]:
    """Scalar summaries for logs / docs (all derived from the calibration file)."""
    rho, m, pos = cal["rho"], cal["second_moment"], cal["pos_second_moment"]
    s = min(svd_components, rho.numel())
    total_pos = float(pos.sum())
    return {
        "num_images": float(cal["meta"].get("num_images", 0)),
        "mean_rho_top_s": float(rho[:s].mean()),
        "mean_rho_tail": float(rho[s:].mean()) if rho.numel() > s else float("nan"),
        "frac_positional_energy_in_top_s": float(pos[:s].sum() / max(total_pos, 1e-12)),
        "frac_feature_energy_in_top_s": float(m[:s].sum() / m.sum().clamp_min(1e-12)),
        "frac_feature_energy_positional": float(pos.sum() / m.sum().clamp_min(1e-12)),
    }
