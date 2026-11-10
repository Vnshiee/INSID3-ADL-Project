"""Shared fixtures: a tiny deterministic stand-in for DINOv3 (CPU, no weights).

FakeEncoder mimics the only API INSID3 uses — ``get_intermediate_layers(x, n=1,
reshape=True)`` returning ``[ (B, C, h, w) ]`` with patch size 16 — and adds a
learned-free *positional* term so that a positional subspace actually exists.
It is NOT a model of DINOv3 quality; it only exercises shapes and code paths.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMAGE_SIZE = 128          # -> 8 x 8 token grid
CHANNELS = 32


class FakeEncoder(nn.Module):
    def __init__(self, channels: int = CHANNELS, image_size: int = IMAGE_SIZE, pos_scale: float = 1.5):
        super().__init__()
        g = torch.Generator().manual_seed(1234)
        h = image_size // 16
        self.proj = nn.Conv2d(3, channels, kernel_size=16, stride=16)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g) * 0.05)
            self.proj.bias.copy_(torch.randn(channels, generator=g) * 0.1)
        # Low-rank smooth positional field: 3 directions modulated by x/y coordinates.
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, h), indexing="ij")
        basis = torch.linalg.qr(torch.randn(channels, 3, generator=g))[0]          # (C, 3)
        field = torch.stack([xs, ys, xs * ys])                                      # (3, h, h)
        self.register_buffer("pos", pos_scale * torch.einsum("ck,khw->chw", basis, field)[None])

    def get_intermediate_layers(self, x, n=1, reshape=True):
        f = torch.tanh(self.proj(x)) + self.pos
        return [f]


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return FakeEncoder().eval()


def make_image(color_box=(0.9, 0.1, 0.1), box=(32, 32, 80, 80), bg=(0.2, 0.5, 0.3), seed=0):
    """(1, 3, S, S) ImageNet-normalised image with a coloured square on a textured background."""
    g = torch.Generator().manual_seed(seed)
    img = torch.tensor(bg).view(3, 1, 1).repeat(1, IMAGE_SIZE, IMAGE_SIZE)
    img = img + 0.05 * torch.randn(img.shape, generator=g)
    y0, x0, y1, x1 = box
    img[:, y0:y1, x0:x1] = torch.tensor(color_box).view(3, 1, 1)
    img[:, y0:(y0 + y1) // 2, x0:x1] *= 0.7               # two "parts" inside the object
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((img.clamp(0, 1) - mean) / std).unsqueeze(0)


def make_mask(box=(32, 32, 80, 80)):
    m = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def episode():
    """1-shot episode: reference square at one place, target square elsewhere."""
    ref = make_image(box=(16, 16, 64, 64), seed=1)
    tgt = make_image(box=(56, 48, 112, 112), seed=2)
    return ref, make_mask((16, 16, 64, 64)).unsqueeze(0), tgt
