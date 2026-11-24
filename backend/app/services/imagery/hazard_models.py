"""Network used by both trained hazard-imagery models (PyTorch).

- flood  : Sentinel-1 radar (VV, VH)            -> water probability per 10 m pixel
- lightning: infrared cloud-top temperature x2  -> flash probability per 8 km cell

One small U-Net serves both: the tasks are dense per-pixel classification
from a few physical channels with a few dozen (flood) to a few dozen-thousand
(lightning) labelled grid cells, so a compact encoder/decoder with skip
connections is the standard, sufficient choice -- a large pretrained
backbone would overfit 80 radar chips and expects 3-channel photographs.

GroupNorm (not BatchNorm): batches are small and inference is one image at a
time, so statistics must not depend on the batch. Any image size works -- it
is padded to a multiple of 2**depth internally and cropped back.

Import this module only where PyTorch is allowed (the inference worker and
the training scripts) -- see nowcast_worker.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

ARCHITECTURE_NAME = "SmallUNet (GroupNorm, SiLU)"


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(_groups(cout), cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(_groups(cout), cout), nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class SmallUNet(nn.Module):
    """`out_stride=1`: logits at input resolution. `out_stride=2`: logits at
    half resolution (the decoder stops one level early) -- used for
    lightning, whose labels live on a grid twice as coarse as the imagery."""

    def __init__(self, in_channels: int, base: int = 24, depth: int = 4, out_stride: int = 1, dropout: float = 0.1):
        super().__init__()
        if out_stride not in (1, 2):
            raise ValueError("out_stride must be 1 or 2")
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.in_channels, self.base, self.depth, self.out_stride, self.dropout = in_channels, base, depth, out_stride, dropout
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [ConvBlock(in_channels, widths[0])] + [ConvBlock(widths[i - 1], widths[i], dropout if i >= 2 else 0.0) for i in range(1, depth + 1)]
        )
        stop = out_stride - 1  # decoder levels run from depth-1 down to `stop`
        self.decoders = nn.ModuleList([ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in range(depth - 1, stop - 1, -1)])
        self.head = nn.Conv2d(widths[stop], 1, 1)

    def config(self) -> dict:
        return {"in_channels": self.in_channels, "base": self.base, "depth": self.depth,
                "out_stride": self.out_stride, "dropout": self.dropout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        multiple = 2 ** self.depth
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1))
        logits = self.head(x)
        out_h, out_w = -(-height // self.out_stride), -(-width // self.out_stride)
        return logits[..., :out_h, :out_w]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy + soft Dice, both restricted to `valid` pixels.

    BCE alone under-weights the rare positive class (11 % water, ~1 %
    lightning); Dice alone is unstable on images with no positives. Their
    sum is the usual remedy."""
    valid = valid.float()
    n = valid.sum().clamp(min=1.0)
    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=weight)
    bce = (bce * valid).sum() / n
    prob = torch.sigmoid(logits) * valid
    tgt = target * valid
    dice = 1.0 - (2.0 * (prob * tgt).sum() + 1.0) / (prob.sum() + tgt.sum() + 1.0)
    return bce + dice
