"""The current code must reproduce the stored golden outputs (see tests/golden_cases.py).

How strictly:
  * masks, labels, counts (non-float tensors): EXACTLY equal.
  * floats: equal up to rtol 1e-3 / atol 1e-4. On the machine where the golden file was made the
    rewrite reproduced every value bit for bit; on another CPU, LAPACK (torch.linalg.svd) and the
    number of threads change the last digits (~1e-5). A real change of the code changes values by
    orders of magnitude more, so this tolerance still catches it.
  * 'cal/template': an SVD may flip the sign of a direction (mathematically the same basis), which
    flips that row of the positional pattern; rows are sign-aligned and the first 8 (well separated)
    directions compared.
  * 'cal/per_image_noise_corr': a correlation stored in half precision; for directions with almost no
    energy it is a ratio of near-zero numbers and meaningless, so only the first 4 directions are compared.
"""

import torch

from golden_cases import GOLDEN, compute_cases


def _difference(key, ref, now):
    """Return None if `now` matches `ref` for this key, else a short description of the mismatch."""
    if ref.shape != now.shape:
        return f"shape {tuple(ref.shape)} vs {tuple(now.shape)}"
    if not ref.is_floating_point():
        return None if torch.equal(ref, now) else "not identical"
    ref, now = ref.double(), now.double()
    if key == "cal/template":
        sign = torch.sign((ref * now).sum(dim=1, keepdim=True))
        sign[sign == 0] = 1
        ref, now = ref[:8], (now * sign)[:8]
    if key == "cal/per_image_noise_corr":
        ok = torch.allclose(ref[:, :4], now[:, :4], rtol=0, atol=1e-2)
        return None if ok else f"max diff {float((ref[:, :4] - now[:, :4]).abs().max()):.2e}"
    if torch.allclose(ref, now, rtol=1e-3, atol=1e-4):
        return None
    return f"max diff {float((ref - now).abs().max()):.2e}"


def test_outputs_identical_to_golden(tmp_path):
    golden = torch.load(GOLDEN, weights_only=False)
    now = compute_cases(str(tmp_path))
    assert set(now) == set(golden), set(now) ^ set(golden)
    problems = {k: d for k in golden if (d := _difference(k, golden[k], now[k])) is not None}
    assert not problems, f"outputs changed: {problems}"
