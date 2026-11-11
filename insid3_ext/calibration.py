"""Task 1 — measuring how "positional" each direction really is (on real images).

THE QUESTION
INSID3 removes 500 directions found on an EMPTY image. On real photos, how much of the energy along
each of those directions is really position, and how much is image content? The topic asks us to
"measure their contribution across different images and feature tokens"; this file does that.

THE MODEL WE ASSUME
Take N real images (not from any test set), resized exactly as at inference. For direction k, image I
and patch position p, the coordinate is c_k(I, p) = u_k . x(I, p). We split it into

        c_k(I, p) = pi_k(p)  +  e_k(I, p)
                    position    content

pi_k(p) depends only on WHERE the patch is (the same for every image); e_k depends on WHAT the image
shows. Position and content are assumed unrelated (uncorrelated).

WHAT WE ESTIMATE (all per direction k)
  template       pi_k(p) = average over images of c_k(I, p), minus its average over positions   (R, P)
  pos energy     v_k     = average over p of pi_k(p)^2, corrected for estimation noise            (R,)
  total energy   m_k     = average over images and positions of c_k^2                             (R,)
  share          rho_k   = v_k / m_k, between 0 and 1                                             (R,)

rho_k is "what fraction of this direction is position" -> the lambda of mode "calibrated".

Why the noise correction: with a finite number of images, the per-position average still contains
some content by chance, which would look like position. That chance part has size
(variance within a position) / N, so we subtract it. (Tested: with no positional signal at all the
corrected rho is ~0 — tests/test_calibration.py.)

EXTRA STATISTICS for the analysis figures
  per_image_second_moment  (N, R)  energy of each image along each direction
  per_image_noise_corr     (N, R)  how closely each image follows the empty image's pattern
  token_energy_mean/std    (P,)    energy each patch position has inside the official 500 directions
  feat_mean, feat_cov      (C,), (C, C)  mean and covariance of all real-image features
                                   (for the 'pca' and 'center' controls)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class PositionalStatsAccumulator:
    """Collects running sums over many images (so all images never have to be in memory at once).

    Sums are kept in float64: we add up hundreds of thousands of small numbers, and float32 would
    lose precision.
    """

    def __init__(self, U: torch.Tensor, noise_template: torch.Tensor, svd_components: int, grid: tuple):
        """
        U               (C, R) the empty-image directions (from estimate_positional_svd)
        noise_template  (R, P) the empty image's own coordinates (for the per-image correlation)
        svd_components  s = 500, the official number of directions (for the per-token energy map)
        grid            (h, w) patch grid; every image must have this grid
        """
        self.U = U.float()
        self.noise_template = noise_template.float()
        self.s = int(svd_components)
        self.grid = tuple(grid)
        R, P = noise_template.shape
        C = U.shape[0]
        dev = U.device

        # Per direction k and position p: sum of c and sum of c^2 over images.
        self.sum_c = torch.zeros(R, P, dtype=torch.float64, device=dev)
        self.sum_c2 = torch.zeros(R, P, dtype=torch.float64, device=dev)
        # Per position: sum (and sum of squares) of the energy inside the official s directions.
        self.sum_e = torch.zeros(P, dtype=torch.float64, device=dev)
        self.sum_e2 = torch.zeros(P, dtype=torch.float64, device=dev)
        # Over all patches of all images: sum of features and of their outer products (-> mean, covariance).
        self.sum_x = torch.zeros(C, dtype=torch.float64, device=dev)
        self.sum_xx = torch.zeros(C, C, dtype=torch.float64, device=dev)
        self.n_tok = 0                                     # number of patches seen
        self.n = 0                                         # number of images seen
        self.per_image_m: List[torch.Tensor] = []
        self.per_image_corr: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        """Add a batch of unit-length feature maps (N, C, h, w)."""
        N, C, h, w = feats.shape
        if (h, w) != self.grid:
            raise ValueError(f"Feature grid {(h, w)} != basis grid {self.grid}")
        X = feats.reshape(N, C, h * w).float().to(self.U.device)      # (N, C, P)

        # Coordinates of every patch along every direction.
        coeff = self.U.T @ X                                          # (N, R, P): c = U^T x
        self.sum_c += coeff.sum(0).double()
        self.sum_c2 += coeff.pow(2).sum(0).double()

        # All patches as rows, for the feature mean / covariance.
        Xt = X.permute(0, 2, 1).reshape(-1, C).double()               # (N*P, C)
        self.sum_x += Xt.sum(0)
        self.sum_xx += Xt.T @ Xt
        self.n_tok += Xt.shape[0]

        # Energy of each patch inside the official s directions (what the hard projection deletes).
        e = coeff[:, : self.s].pow(2).sum(1)                          # (N, P)
        self.sum_e += e.sum(0).double()
        self.sum_e2 += e.pow(2).sum(0).double()

        # Per image: its energy along each direction ...
        self.per_image_m.append(coeff.pow(2).mean(2).half().cpu())    # (N, R)
        # ... and the correlation (over positions) between the image's coordinates and the empty
        # image's pattern: close to 1 means "this image shows the positional pattern clearly".
        cc = coeff - coeff.mean(2, keepdim=True)
        tt = self.noise_template - self.noise_template.mean(1, keepdim=True)          # (R, P)
        corr = (cc * tt[None]).sum(2) / (cc.norm(dim=2) * tt.norm(dim=1)[None]).clamp_min(1e-12)
        self.per_image_corr.append(corr.half().cpu())                 # (N, R)
        self.n += N

    def finalize(self, meta: Optional[Dict[str, Any]] = None, S: Optional[torch.Tensor] = None
                 ) -> Dict[str, Any]:
        """Turn the sums into the statistics described at the top of this file."""
        if self.n < 2:
            raise ValueError("Need at least 2 calibration images.")
        n = float(self.n)

        mean = self.sum_c / n                       # (R, P) average coordinate at each position
        second = self.sum_c2 / n                    # (R, P) average squared coordinate at each position
        grand = mean.mean(1)                        # (R,)   average over all positions
        template = mean - grand[:, None]            # (R, P) pi_k(p): positional pattern, zero mean over p

        # Spread of the coordinate between images at a fixed position (content), unbiased (n / (n-1)).
        within_unbiased = (second - mean.pow(2)).clamp_min(0) * n / (n - 1)
        # Spread of the pattern over positions; still contains a chance part of size within / n ...
        between_naive = template.pow(2).mean(1)                                        # (R,)
        # ... which we subtract (never below 0): this is the positional energy v_k.
        pos_var = (between_naive - within_unbiased.mean(1) / n).clamp_min(0)

        second_moment = second.mean(1)              # (R,) m_k: total energy along direction k
        total_var = second_moment - grand.pow(2)
        # rho_k = positional energy / total energy. We divide by the second moment (not the variance)
        # because the debiaser removes lambda * c with no offset, so the matching optimum uses E[c^2].
        rho = (pos_var / second_moment.clamp_min(1e-12)).clamp(0, 1)

        e_mean = self.sum_e / n                                      # (P,)
        e_std = (self.sum_e2 / n - e_mean.pow(2)).clamp_min(0).sqrt()

        feat_mean = self.sum_x / self.n_tok                          # (C,)
        feat_cov = self.sum_xx / self.n_tok - torch.outer(feat_mean, feat_mean)   # (C, C)

        return {
            "feat_mean": feat_mean.float().cpu(),
            "feat_cov": feat_cov.float().cpu(),
            "meta": dict(meta or {}, num_images=int(self.n), grid=list(self.grid), svd_components=self.s),
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


def summarize_calibration(cal: Dict[str, Any], svd_components: int) -> Dict[str, float]:
    """A few numbers for the report, computed from the calibration file."""
    rho, m, pos = cal["rho"], cal["second_moment"], cal["pos_second_moment"]
    s = min(svd_components, rho.numel())
    total_pos = float(pos.sum())
    return {
        "num_images": float(cal["meta"].get("num_images", 0)),
        # average positional share inside / outside the official 500 directions
        "mean_rho_top_s": float(rho[:s].mean()),
        "mean_rho_tail": float(rho[s:].mean()) if rho.numel() > s else float("nan"),
        # how much of ALL positional energy the official 500 directions contain
        "frac_positional_energy_in_top_s": float(pos[:s].sum() / max(total_pos, 1e-12)),
        # how much of ALL feature energy the official 500 directions contain (= what hard projection deletes)
        "frac_feature_energy_in_top_s": float(m[:s].sum() / m.sum().clamp_min(1e-12)),
        # how much of all feature energy is positional
        "frac_feature_energy_positional": float(pos.sum() / m.sum().clamp_min(1e-12)),
    }
