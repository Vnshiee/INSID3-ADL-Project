"""Task 1 — Adaptive Positional Debiasing.

Original INSID3 (models/insid3.py, ``_build_positional_basis`` / ``_debias_features``)
--------------------------------------------------------------------------------
1. A constant image (zeros after ImageNet normalisation; the paper text says
   "noise image") is encoded once: ``F_noise`` of shape (C, P), P = h*w tokens.
2. Every token is L2-normalised, the token mean is subtracted:  E = F_noise - mean.
3. SVD  E = U Σ Vᵀ.  The positional basis is  B = U[:, :s]  (C × s), s = 500.
4. For every (already L2-normalised) feature x ∈ R^C:
        x̃ = normalize( (I - B Bᵀ) x ).

Generalisation implemented here
-------------------------------
With coefficients c = Uᵣᵀ x ∈ R^r (Uᵣ = first r columns of U) every mode is

        x̃ = normalize( x − Uᵣ (λ ⊙ c) ),        λ ∈ [0, 1]^r  (or token-wise),

so λ_k is *the fraction of component k that is removed*.
    hard        λ_k = 1 for k ≤ s                      (original, exact code path)
    none        λ_k = 0                                 (no debiasing)
    weighted    λ_k = α                                  (fixed weighted projection)
    spectral    λ_k = σ_k^{2γ} / (σ_k^{2γ} + σ_knee^{2γ})  (data-free component-wise; γ→∞ ⇒ hard)
    calibrated  λ_k = clamp(β ρ_k, 0, 1)                 (component-wise MMSE gain, ρ from calibration)
    image       λ_k(I) = clamp(β v_k^pos / m_k(I), 0, 1)  (per-image Wiener gain)
    token       x̃_p = normalize( x_p − γ_p Uᵣ π_p )       (per-token template subtraction)

Derivations are in docs/MATHEMATICAL_FORMULATION.md §2.

Shapes (B = batch of episodes, T = images per episode, C = channels, P = h*w)
    fmaps_norm  (B, T, C, h, w)  L2-normalised over C
    U           (C, R)           orthonormal columns, R = min(C, P)
    S           (R,)             singular values of E (descending)
    coeff       (B*T, r, P)      projection coefficients
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import einops
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PositionalSVD:
    """Full SVD of the centred constant-image features.

    Attributes:
        U:      (C, R) left singular vectors (= paper's right singular vectors of Fᵀ).
        S:      (R,)   singular values.
        coeff:  (R, P) coefficients Uᵀ E of the centred constant-image tokens,
                i.e. the *positional template* seen by the encoder on an empty image.
        grid:   (h, w) token grid the SVD was computed on.
    """
    U: torch.Tensor
    S: torch.Tensor
    coeff: torch.Tensor
    grid: tuple

    @property
    def num_tokens(self) -> int:
        return int(self.grid[0] * self.grid[1])


@torch.no_grad()
def estimate_positional_svd(encoder: torch.nn.Module, image_size: int, device: str) -> PositionalSVD:
    """Re-implementation of ``INSID3._build_positional_basis`` that keeps the full SVD.

    The operations are line-for-line those of the official method (same input image,
    same normalisation, same centring, same ``torch.linalg.svd`` call), so that
    ``U[:, :s]`` equals the official basis. This is verified by
    ``tests/test_baseline_equivalence.py``.
    """
    from torchvision.transforms.functional import normalize
    noise_img = normalize(
        torch.zeros(1, 3, image_size, image_size),
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ).to(device)
    noise_fmaps = encoder.to(device).get_intermediate_layers(noise_img, n=1, reshape=True)[0]
    noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)          # (1, C, h, w)
    h, w = noise_fmaps.shape[-2:]

    E = einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)')    # (C, P)
    E = E - E.mean(dim=1, keepdim=True)
    U, S, _ = torch.linalg.svd(E, full_matrices=False)          # U: (C, R), S: (R,)
    coeff = U.T @ E                                              # (R, P) = diag(S) Vᵀ
    return PositionalSVD(U=U, S=S, coeff=coeff, grid=(int(h), int(w)))


# ───────────────────────────── gain schedules (static λ) ─────────────────────────────

def weighted_gains(rank: int, alpha: float) -> torch.Tensor:
    """λ_k = α for k = 1..rank (α = 1 recovers the hard projection)."""
    return torch.full((rank,), float(alpha))


def spectral_gains(S: torch.Tensor, rank: int, knee_rank: int, sharpness: float) -> torch.Tensor:
    """Data-free component-wise attenuation.

    λ_k = σ_k^{2γ} / (σ_k^{2γ} + σ_knee^{2γ}) = sigmoid(2γ (log σ_k − log σ_knee)).

    Interpretation: Wiener gain under the assumption that semantic variance is isotropic
    with level σ_knee² while positional variance along u_k is ∝ σ_k². λ = 0.5 at the knee;
    as γ → ∞ the schedule becomes the hard rank-``knee_rank`` truncation of INSID3.
    """
    if knee_rank > S.numel():
        raise ValueError(f"knee_rank={knee_rank} exceeds the number of singular values ({S.numel()})")
    s = S[:rank].double().clamp_min(1e-12)
    knee = S[knee_rank - 1].double().clamp_min(1e-12)
    return torch.sigmoid(2.0 * sharpness * (s.log() - knee.log())).float()


def calibrated_gains(rho: torch.Tensor, rank: int, strength: float) -> torch.Tensor:
    """λ_k = clamp(β ρ_k, 0, 1), ρ_k = positional share of the 2nd moment of c_k (calibration.py)."""
    return (strength * rho[:rank].float()).clamp(0.0, 1.0)


# ───────────────────────────── calibration I/O ─────────────────────────────

def load_calibration(path: str, model_size: Optional[str] = None,
                     image_size: Optional[int] = None) -> Dict[str, Any]:
    """Load a calibration file produced by ``scripts/calibrate_positional.py``.

    Raises a clear error if it was computed for a different backbone or resolution,
    because U, ρ and the templates are only meaningful for the same token grid.
    """
    cal = torch.load(path, map_location="cpu", weights_only=False)
    meta = cal.get("meta", {})
    if model_size is not None and meta.get("model_size") != model_size:
        raise ValueError(f"Calibration {path} was computed for model_size={meta.get('model_size')}, "
                         f"but the model uses {model_size}.")
    if image_size is not None and meta.get("image_size") != image_size:
        raise ValueError(f"Calibration {path} was computed for image_size={meta.get('image_size')}, "
                         f"but the model uses {image_size}.")
    for key in ("U", "S", "rho", "pos_second_moment", "template"):
        if key not in cal:
            raise ValueError(f"Calibration file {path} is missing key '{key}'.")
    return cal


def subspace_agreement(U_a: torch.Tensor, U_b: torch.Tensor, k: int) -> float:
    """Mean squared cosine of principal angles between span(U_a[:, :k]) and span(U_b[:, :k]).

    1.0 means identical subspaces; used to check a calibration matches the model's basis.
    """
    A, B = U_a[:, :k].double(), U_b[:, :k].double()
    return float((A.T @ B).pow(2).sum() / k)


# ───────────────────────────── the debiaser ─────────────────────────────

class PositionalDebiaser:
    """Applies one of the debiasing modes to L2-normalised feature maps.

    Use :meth:`from_config` to construct. Calling the object maps
    (B, T, C, h, w) → (B, T, C, h, w), L2-normalised over C (like the original).
    """

    STATIC_MODES = ("none", "weighted", "spectral", "calibrated")

    def __init__(self, mode: str, U: torch.Tensor, *,
                 hard_basis: Optional[torch.Tensor] = None,
                 gains: Optional[torch.Tensor] = None,
                 pos_second_moment: Optional[torch.Tensor] = None,
                 template: Optional[torch.Tensor] = None,
                 strength: float = 1.0,
                 token_gain: str = "least_squares",
                 max_gain: float = 2.0,
                 grid: Optional[tuple] = None):
        self.mode = mode
        self.U = U                          # (C, r) basis used by the soft modes
        self.hard_basis = hard_basis        # (C, s) official basis, used by mode='hard'
        self.gains = gains                  # (r,)   static λ
        self.pos_second_moment = pos_second_moment  # (r,) positional energy per component
        self.template = template            # (r, P) positional template coefficients π_p
        self.strength = float(strength)
        self.token_gain = token_gain
        self.max_gain = float(max_gain)
        self.grid = grid
        self.last_stats: Dict[str, float] = {}

    @property
    def rank(self) -> int:
        return 0 if self.U is None else int(self.U.shape[1])

    # ---------------- construction ----------------
    @classmethod
    def from_config(cls, cfg, svd: PositionalSVD, svd_components: int,
                    calibration: Optional[Dict[str, Any]] = None) -> "PositionalDebiaser":
        """Build from a :class:`insid3_ext.config.DebiasConfig`.

        Args:
            cfg: DebiasConfig.
            svd: full SVD of the constant image (from the model's encoder).
            svd_components: s of the official hard projection.
            calibration: dict from :func:`load_calibration` (required by some modes).
        """
        mode = cfg.mode
        hard_basis = svd.U[:, :svd_components].contiguous()
        uses_cal = (mode == "calibrated"
                    or (mode == "image" and cfg.variance_source == "calibration")
                    or (mode == "token" and cfg.template_source == "calibration"))
        if uses_cal and calibration is None:
            raise ValueError(f"debias.mode={mode} requires a calibration file.")

        if uses_cal:
            U_full, S_full = calibration["U"].to(svd.U), calibration["S"].to(svd.S)
            k = min(svd_components, U_full.shape[1], svd.U.shape[1])
            if k > 0:
                agree = subspace_agreement(U_full, svd.U, k)
                if agree < 0.99:
                    warnings.warn(f"Calibration basis differs from the model's positional basis "
                                  f"(subspace agreement {agree:.3f} over the top {k} components). "
                                  f"Was it computed with the same weights/resolution?")
        else:
            U_full, S_full = svd.U, svd.S

        R = U_full.shape[1]
        default_rank = svd_components if mode in ("hard", "weighted") else R
        rank = min(cfg.rank or default_rank, R)
        U = U_full[:, :rank].contiguous()

        kw: Dict[str, Any] = dict(hard_basis=hard_basis, strength=cfg.strength,
                                  token_gain=cfg.token_gain, max_gain=cfg.max_gain, grid=svd.grid)
        if mode == "hard":
            pass
        elif mode == "none":
            kw["gains"] = torch.zeros(rank)
        elif mode == "weighted":
            kw["gains"] = weighted_gains(rank, cfg.alpha)
        elif mode == "spectral":
            kw["gains"] = spectral_gains(S_full, rank, cfg.knee_rank, cfg.sharpness)
        elif mode == "calibrated":
            kw["gains"] = calibrated_gains(calibration["rho"], rank, cfg.strength)
        elif mode == "image":
            if cfg.variance_source == "calibration":
                kw["pos_second_moment"] = calibration["pos_second_moment"][:rank].float()
            else:  # per-token energy of the centred constant image: σ_k² / P
                kw["pos_second_moment"] = (svd.S[:rank].float() ** 2) / svd.num_tokens
        elif mode == "token":
            if cfg.template_source == "calibration":
                kw["template"] = calibration["template"][:rank].float()
                kw["grid"] = tuple(calibration["meta"]["grid"])
            else:
                kw["template"] = svd.coeff[:rank].float()
        else:
            raise ValueError(f"Unknown debias mode: {mode}")
        return cls(mode, U, **kw)

    # ---------------- application ----------------
    def static_gains(self) -> Optional[torch.Tensor]:
        """λ for the static modes (for plotting); None for image/token/hard."""
        return None if self.gains is None else self.gains.clone()

    def __call__(self, fmaps_norm: torch.Tensor) -> torch.Tensor:
        """Debias L2-normalised feature maps.

        Args:
            fmaps_norm: (B, T, C, h, w), L2-normalised along C.
        Returns:
            (B, T, C, h, w), L2-normalised along C.
        """
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)                     # (N, C, P)

        if self.mode == "hard":
            # Verbatim official operation (models/insid3.py:_debias_features).
            basis = self.hard_basis.to(device=X.device, dtype=X.dtype)
            P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
            X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
            return F.normalize(X_deb, p=2, dim=2)

        U = self.U.to(device=X.device, dtype=X.dtype)                # (C, r)
        coeff = torch.einsum('cr,ncp->nrp', U, X)                    # (N, r, P)

        if self.mode in self.STATIC_MODES:
            lam = self.gains.to(device=X.device, dtype=X.dtype)      # (r,)
            removed = coeff * lam[None, :, None]                     # (N, r, P)
            self.last_stats = {"mean_lambda": float(self.gains.mean())}

        elif self.mode == "image":
            # m_k(I) = mean_p c_k(p)^2  (second moment over the tokens of image I), fp32 stats.
            m = coeff.float().pow(2).mean(dim=2)                     # (N, r)
            v_pos = self.pos_second_moment.to(device=X.device)       # (r,)
            lam = (self.strength * v_pos[None] / m.clamp_min(1e-12)).clamp(0.0, 1.0)  # (N, r)
            removed = coeff * lam.to(X.dtype)[:, :, None]
            self.last_stats = {"mean_lambda": float(lam.mean())}

        elif self.mode == "token":
            if self.template.shape[1] != H * W:
                raise ValueError(f"Token template has {self.template.shape[1]} positions but the "
                                 f"feature grid is {H}x{W}; use the same image_size as calibration.")
            pi = self.template.to(device=X.device)                    # (r, P)
            if self.token_gain == "fixed":
                gamma = torch.full((X.shape[0], H * W), self.strength, device=X.device)
            else:
                # Least-squares amplitude of the template in each token, γ_p = <c_p, π_p>/||π_p||².
                num = torch.einsum('nrp,rp->np', coeff.float(), pi)  # (N, P)
                den = pi.pow(2).sum(dim=0).clamp_min(1e-12)          # (P,)
                gamma = (self.strength * num / den[None]).clamp(0.0, self.max_gain)
            removed = pi.to(X.dtype)[None] * gamma.to(X.dtype)[:, None, :]  # (N, r, P)
            self.last_stats = {"mean_gamma": float(gamma.mean())}
        else:
            raise ValueError(f"Unknown debias mode: {self.mode}")

        X_deb = X - torch.einsum('cr,nrp->ncp', U, removed)          # (N, C, P)
        return F.normalize(X_deb.reshape(B, T, C, H, W), p=2, dim=2)
