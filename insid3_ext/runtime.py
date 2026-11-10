"""Runtime helpers: device checks, seeding, environment report, timing, memory."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict

import numpy as np
import torch


def resolve_device(requested: str) -> str:
    """Validate the requested device and fail early with an actionable message."""
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "model.device is 'cuda' but torch.cuda.is_available() is False.\n"
                f"  torch={torch.__version__}, built with CUDA={torch.version.cuda}\n"
                "  Check `nvidia-smi`, the driver/CUDA compatibility and that a CUDA build of "
                "PyTorch is installed (see docs/SERVER_SETUP.md). For a CPU smoke test pass "
                "--set model.device=cpu (very slow at 1024 px).")
        idx = int(requested.split(":")[1]) if ":" in requested else 0
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {requested} but only {torch.cuda.device_count()} GPU(s) visible.")
    elif requested != "cpu" and requested != "mps":
        raise RuntimeError(f"Unsupported device '{requested}'.")
    return requested


def set_seed(seed: int) -> None:
    """Identical to the official inference scripts (seed + cudnn flags + TF32 on Ampere+)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def environment_report() -> Dict[str, Any]:
    """Versions and hardware, saved next to every run for reproducibility."""
    rep: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hostname": platform.node(),
    }
    try:
        import sklearn
        rep["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        rep["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rep["git_commit"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=5).stdout.strip() or None
    except Exception:
        rep["git_commit"] = None
    return rep


class CudaTimer:
    """Wall-clock timer that synchronises CUDA so GPU work is included."""

    def __init__(self, device: str):
        self.sync = device.startswith("cuda") and torch.cuda.is_available()

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync:
            torch.cuda.synchronize()
        self.ms = 1000.0 * (time.perf_counter() - self.t0)
        return False


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return float("nan")
