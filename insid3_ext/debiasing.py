"""Task 1 — Adaptive Positional Debiasing.

WHAT THE ORIGINAL INSID3 DOES (models/insid3.py: _build_positional_basis, _debias_features)
------------------------------------------------------------------------------------------
1. Feed an "empty" image to DINOv3: an all-zero image after ImageNet normalisation, i.e. a picture
   with no content at all. Whatever structure its features still have can only come from the
   *position* of each patch, not from what the patch shows.
2. L2-normalise every patch feature and subtract the mean over patches -> matrix E (C x P).
3. SVD:  E = U S V^T.  The first s = 500 columns of U span the "positional subspace" B.
4. For every real feature x:   x_debiased = normalize( (I - B B^T) x )
   i.e. delete everything x has inside B ("hard projection") and rescale to length 1.

WHAT WE CHANGE
--------------
Instead of deleting everything inside B, we delete only a FRACTION of each direction.
Write c = U^T x (the coordinates of x along the directions u_1, u_2, ...). Then every mode is

        x_debiased = normalize( x - U (lambda * c) )        with 0 <= lambda_k <= 1

lambda_k = 1 means "delete direction k completely", lambda_k = 0 means "keep it".
The hard projection is the special case lambda_k = 1 for k <= 500 and 0 afterwards.

The modes only differ in HOW lambda is chosen. There are four kinds of operation:

  kind "official"  mode hard           exactly the official code (kept as the baseline)
  kind "static"    mode none           lambda_k = 0                      (no debiasing)
                   mode weighted       lambda_k = alpha                  (fixed weighted projection)
                   mode spectral       lambda_k from the singular values (soft rank cut-off, data-free)
                   mode calibrated     lambda_k = beta * rho_k           (rho_k = measured positional share)
                   mode rho_threshold  lambda_k = 1 if rho_k >= tau else 0  (control)
                   mode random         lambda = 1 on a RANDOM s-dim subspace        (control)
                   mode pca            lambda = 1 on the top-s PCA directions of real images (control)
  kind "image"     mode image          lambda_k depends on how much energy THIS image has along u_k
  kind "token"     mode token          subtract the average positional pattern at each patch position

"static" means lambda is a fixed vector, the same for every image and every patch.
The formulas are derived in docs/2_method.md section 2.1.

TENSOR SHAPES used below
  fmaps   (B, T, C, h, w)   B episodes, T images per episode, C channels, h x w patches
  X       (N, C, P)         N = B*T images, P = h*w patches
  U       (C, r)            r orthonormal directions (columns)
  coeff   (N, r, P)         coordinate of every patch along every direction
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import einops
import torch
import torch.nn.functional as F

# ImageNet mean/std: the same normalisation INSID3 applies to every input image.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════════════════
# 1. The positional SVD (identical to the official basis, but we keep all of it)
# ════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionalSVD:
    """Everything the SVD of the empty image gives us.

    U      (C, R)  all directions (the official code only keeps U[:, :500])
    S      (R,)    their singular values = how strongly each direction varies over positions
    coeff  (R, P)  the empty image's coordinates along each direction at each patch position:
                   this is the "positional pattern" the encoder produces when there is no content
    grid   (h, w)  patch grid the SVD was computed on (the patterns only fit this grid)
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
    """Same steps as INSID3._build_positional_basis (lines 246-261), but returns the full SVD.

    Every line below repeats the official code so that U[:, :s] is bit-identical to the official
    basis (checked by tests/test_baseline_equivalence.py::test_positional_basis_identical).
    """
    from torchvision.transforms.functional import normalize

    # The "empty" image: all zeros, then ImageNet-normalised exactly like a real input.
    noise_img = normalize(torch.zeros(1, 3, image_size, image_size),
                          mean=IMAGENET_MEAN, std=IMAGENET_STD).to(device)

    # Encode it. As in the official code there is no autocast here (full fp32 precision).
    noise_fmaps = encoder.to(device).get_intermediate_layers(noise_img, n=1, reshape=True)[0]

    # Unit length per patch, because INSID3 compares features by cosine similarity.
    noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)                 # (1, C, h, w)
    h, w = noise_fmaps.shape[-2:]

    # One column per patch, then remove the average patch: what remains is how patches DIFFER
    # from each other, and on an empty image they can only differ because of their position.
    E = einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)')           # (C, P)
    E = E - E.mean(dim=1, keepdim=True)

    # SVD sorts the directions from "varies most with position" to "varies least".
    U, S, _ = torch.linalg.svd(E, full_matrices=False)                 # U (C, R), S (R,)

    # ORIGINAL ended here with:   U, _, _ = torch.linalg.svd(...);  return U[:, :self.svd_components]
    # OURS keeps S and all of U, and additionally:
    # Coordinates of the empty image along every direction (= S * V^T); used by mode "token".
    coeff = U.T @ E                                                     # (R, P)
    return PositionalSVD(U=U, S=S, coeff=coeff, grid=(int(h), int(w)))


# ════════════════════════════════════════════════════════════════════════════════════════
# 2. How much to remove per direction (the lambda vectors of the "static" modes)
# ════════════════════════════════════════════════════════════════════════════════════════

def weighted_gains(rank: int, alpha: float) -> torch.Tensor:
    """Mode weighted: remove the same fraction alpha of every direction (alpha = 1 is the official method)."""
    return torch.full((rank,), float(alpha))


def spectral_gains(S: torch.Tensor, rank: int, knee_rank: int, sharpness: float) -> torch.Tensor:
    """Mode spectral: a SOFT version of "keep only the first knee_rank directions".

    lambda_k = sigma_k^(2g) / (sigma_k^(2g) + sigma_knee^(2g)) = sigmoid(2g * (log sigma_k - log sigma_knee))

    - directions much stronger than the knee direction get lambda ~ 1 (removed),
      much weaker ones get lambda ~ 0 (kept), the knee itself gets exactly 0.5;
    - g = sharpness: g = 1 is the classical Wiener filter shape; large g makes the curve a step,
      i.e. the official hard cut at k = knee_rank (so the original is a limit case of this mode).
    """
    if knee_rank > S.numel():
        raise ValueError(f"knee_rank={knee_rank} exceeds the number of singular values ({S.numel()})")
    # float64 + clamp: log of a (near-)zero singular value would be -inf.
    s = S[:rank].double().clamp_min(1e-12)
    knee = S[knee_rank - 1].double().clamp_min(1e-12)     # knee_rank is 1-based, tensors are 0-based
    return torch.sigmoid(2.0 * sharpness * (s.log() - knee.log())).float()


def calibrated_gains(rho: torch.Tensor, rank: int, strength: float) -> torch.Tensor:
    """Mode calibrated: lambda_k = beta * rho_k, cut to [0, 1].

    rho_k (from calibration.py) is the measured share of direction k's energy on real images that is
    positional. Removing exactly that share is the best linear choice (MMSE, see the math doc);
    beta = strength = 1 is that optimum, beta > 1 removes more than the measured share.
    """
    return (strength * rho[:rank].float()).clamp(0.0, 1.0)


# ════════════════════════════════════════════════════════════════════════════════════════
# 3. Calibration file helpers
# ════════════════════════════════════════════════════════════════════════════════════════

def load_calibration(path: str, model_size: Optional[str] = None,
                     image_size: Optional[int] = None) -> Dict[str, Any]:
    """Read a file written by scripts/calibrate_positional.py and check it fits the model.

    The directions and patterns only make sense for the same backbone and the same resolution
    (a different resolution means a different patch grid), so a mismatch is an error, not a warning.
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
    """How similar are the spaces spanned by the first k columns of U_a and U_b? (1.0 = identical)

    We compare spaces, not individual columns, because an SVD may flip the sign of a column.
    """
    A, B = U_a[:, :k].double(), U_b[:, :k].double()
    return float((A.T @ B).pow(2).sum() / k)


# ════════════════════════════════════════════════════════════════════════════════════════
# 4. The debiaser
# ════════════════════════════════════════════════════════════════════════════════════════

class PositionalDebiaser:
    """Applies one debiasing mode. Build it with PositionalDebiaser.from_config(...).

    Calling it maps (B, T, C, h, w) -> (B, T, C, h, w); output vectors have unit length.
    """

    def __init__(self, kind: str, U: torch.Tensor, *, gains=None, positional_energy=None,
                 template=None, strength: float = 1.0, token_gain: str = "least_squares",
                 max_gain: float = 2.0, mode: Optional[str] = None):
        self.kind = kind                            # "official" | "static" | "image" | "token"
        self.mode = mode or kind                    # the config name, only for logging
        self.U = U                                  # (C, r) directions we act on
        self.gains = gains                          # (r,)  lambda of the static modes
        self.positional_energy = positional_energy  # (r,)  expected positional energy (mode image)
        self.template = template                    # (r, P) positional pattern per position (mode token)
        self.strength = float(strength)             # beta (modes image and token)
        self.token_gain = token_gain                # "fixed" | "least_squares" (mode token)
        self.max_gain = float(max_gain)             # upper limit of the token gain
        self.last_stats: Dict[str, float] = {}      # mean lambda/gamma of the last call, for the logs

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    # ───────────────────────────── construction ─────────────────────────────

    @classmethod
    def from_config(cls, cfg, svd: PositionalSVD, svd_components: int,
                    calibration: Optional[Dict[str, Any]] = None) -> "PositionalDebiaser":
        """Turn a DebiasConfig into a debiaser.

        cfg             DebiasConfig (insid3_ext/config.py)
        svd             the empty-image SVD of THIS model's encoder
        svd_components  s of the official method (500 for ViT-L)
        calibration     dict from load_calibration (needed by some modes)
        """
        from .config import needs_calibration
        mode = cfg.mode
        common = dict(strength=cfg.strength, token_gain=cfg.token_gain, max_gain=cfg.max_gain, mode=mode)

        # Step 1 — the official case needs nothing else: just the first s directions of the model's SVD.
        if mode == "hard":
            return cls("official", svd.U[:, :svd_components].contiguous(), **common)

        # Step 2 — which set of directions do we work in?
        # Modes that read the calibration file use the directions stored IN that file, so that the
        # measured rho / patterns and the directions belong together. The file was computed with the
        # same procedure, so its directions should match the model's; we check and warn otherwise.
        if needs_calibration(cfg):
            if calibration is None:
                raise ValueError(f"debias.mode={mode} requires a calibration file.")
            U_all, S_all = calibration["U"].to(svd.U), calibration["S"].to(svd.S)
            k = min(svd_components, U_all.shape[1], svd.U.shape[1])
            if k > 0:
                agree = subspace_agreement(U_all, svd.U, k)
                if agree < 0.99:
                    warnings.warn(f"Calibration basis differs from the model's positional basis "
                                  f"(subspace agreement {agree:.3f} over the top {k} components). "
                                  f"Was it computed with the same weights/resolution?")
        else:
            U_all, S_all = svd.U, svd.S

        # Step 3 — how many directions? "weighted" acts on the official s directions (so alpha = 1 is
        # exactly the official method); the other modes use ALL directions and let lambda decide.
        R = U_all.shape[1]
        default_rank = svd_components if mode == "weighted" else R
        rank = min(cfg.rank or default_rank, R)
        U = U_all[:, :rank].contiguous()

        # Step 4 — the lambda (or pattern) of each mode.
        if mode == "none":
            return cls("static", U, gains=torch.zeros(rank), **common)
        if mode == "weighted":
            return cls("static", U, gains=weighted_gains(rank, cfg.alpha), **common)
        if mode == "spectral":
            return cls("static", U, gains=spectral_gains(S_all, rank, cfg.knee_rank, cfg.sharpness), **common)
        if mode == "calibrated":
            return cls("static", U, gains=calibrated_gains(calibration["rho"], rank, cfg.strength), **common)
        if mode == "rho_threshold":
            # Control: delete completely the directions that are measurably positional, keep the rest.
            gains = (calibration["rho"][:rank] >= cfg.rho_threshold).float()
            return cls("static", U, gains=gains, **common)
        if mode in ("random", "pca"):
            # Controls: delete s directions that are NOT the positional ones, to see whether the official
            # gain comes from "positional" directions specifically.
            s = min(cfg.rank or svd_components, svd.U.shape[0])
            if mode == "random":
                # s random orthonormal directions (QR of a Gaussian matrix), reproducible via control_seed.
                g = torch.Generator().manual_seed(int(cfg.control_seed))
                basis = torch.linalg.qr(torch.randn(svd.U.shape[0], s, generator=g, dtype=torch.float64))[0]
            else:
                # The s directions in which REAL image features vary most (largest eigenvalues of
                # their covariance matrix, measured during calibration).
                if "feat_cov" not in calibration:
                    raise ValueError("debias.mode=pca needs a calibration file with 'feat_cov'; re-run "
                                     "scripts/calibrate_positional.py with the revised code.")
                evals, evecs = torch.linalg.eigh(calibration["feat_cov"].double())
                basis = evecs[:, torch.argsort(evals, descending=True)[:s]]
            return cls("static", basis.to(svd.U.dtype).contiguous(), gains=torch.ones(s), **common)
        if mode == "image":
            # Expected positional energy per direction: measured on real images (calibration), or taken
            # from the empty image (each of its P patches carries sigma_k^2 / P of the energy along u_k).
            if cfg.variance_source == "calibration":
                energy = calibration["pos_second_moment"][:rank].float()
            else:
                energy = (svd.S[:rank].float() ** 2) / svd.num_tokens
            return cls("image", U, positional_energy=energy, **common)
        if mode == "token":
            # The average positional pattern at each patch position: measured on real images
            # (calibration), or the empty image's own pattern.
            if cfg.template_source == "calibration":
                template = calibration["template"][:rank].float()
            else:
                template = svd.coeff[:rank].float()
            return cls("token", U, template=template, **common)
        raise ValueError(f"Unknown debias mode: {mode}")

    # ───────────────────────────── application ─────────────────────────────

    def __call__(self, fmaps_norm: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Debias unit-length feature maps (B, T, C, h, w).

        normalize=False returns the vectors before rescaling to length 1; only the analysis script
        uses that (to measure how much energy was removed). Inside the model it is always True.
        """
        B, T, C, H, W = fmaps_norm.shape
        X = fmaps_norm.reshape(B * T, C, H * W)                       # (N, C, P): one column per patch

        if self.kind == "official":
            # Exactly the official lines (models/insid3.py 263-271): build I - B B^T and apply it.
            basis = self.U.to(device=X.device, dtype=X.dtype)
            P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
            X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
            return F.normalize(X_deb, p=2, dim=2) if normalize else X_deb

        # All other kinds work with the coordinates c = U^T x of every patch.
        U = self.U.to(device=X.device, dtype=X.dtype)                 # (C, r)
        coeff = U.T @ X                                               # (N, r, P): c = U^T x for every patch

        if self.kind == "static":
            # The same lambda for every image and patch: remove lambda_k * c_k along each direction.
            lam = self.gains.to(device=X.device, dtype=X.dtype)       # (r,)
            removed = coeff * lam[None, :, None]                      # (N, r, P)
            self.last_stats = {"mean_lambda": float(self.gains.mean())}

        elif self.kind == "image":
            # Energy this image has along each direction: average of c_k^2 over its patches.
            # (Statistics in fp32 for accuracy.)
            energy = coeff.float().pow(2).mean(dim=2)                 # (N, r)
            expected = self.positional_energy.to(device=X.device)     # (r,)
            # Wiener rule: lambda = positional energy / total energy. If this image has much more energy
            # than position alone explains, the extra is content -> lambda small -> keep the direction.
            lam = (self.strength * expected[None] / energy.clamp_min(1e-12)).clamp(0.0, 1.0)  # (N, r)
            removed = coeff * lam.to(X.dtype)[:, :, None]
            self.last_stats = {"mean_lambda": float(lam.mean())}

        elif self.kind == "token":
            if self.template.shape[1] != H * W:
                raise ValueError(f"Token template has {self.template.shape[1]} positions but the "
                                 f"feature grid is {H}x{W}; use the same image_size as calibration.")
            pattern = self.template.to(device=X.device)               # (r, P): pattern at each position p
            if self.token_gain == "fixed":
                # Subtract the pattern with the same weight beta at every patch.
                gamma = torch.full((X.shape[0], H * W), self.strength, device=X.device)
            else:
                # Subtract it as strongly as this patch actually shows it (least squares fit):
                # gamma_p = <c_p, pattern_p> / |pattern_p|^2, limited to [0, max_gain].
                num = (coeff.float() * pattern).sum(dim=1)                        # (N, P): <c_p, pattern_p>
                den = pattern.pow(2).sum(dim=0).clamp_min(1e-12)                  # (P,)
                gamma = (self.strength * num / den[None]).clamp(0.0, self.max_gain)
            removed = pattern.to(X.dtype)[None] * gamma.to(X.dtype)[:, None, :]  # (N, r, P)
            self.last_stats = {"mean_gamma": float(gamma.mean())}

        else:
            raise ValueError(f"Unknown debiaser kind: {self.kind}")

        # Go back from coordinates to feature space and subtract: x - U * removed.
        X_deb = X - U @ removed                                       # (N, C, P): x - U * removed
        X_deb = X_deb.reshape(B, T, C, H, W)
        # Rescale to unit length, as the official method does after its projection.
        return F.normalize(X_deb, p=2, dim=2) if normalize else X_deb
