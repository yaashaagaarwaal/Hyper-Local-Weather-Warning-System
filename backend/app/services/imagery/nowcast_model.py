"""The network itself: a small two-scale ConvLSTM encoder with a U-Net-style
decoder that predicts, for each lead time, a *change* relative to the last
observed frame plus a per-pixel uncertainty.

Why this shape (see docs/imagery_model_report.md for the data-driven case):
- The dataset is a genuine half-hourly image sequence with no labels, so the
  only supervised signal available is "what did the satellite see next" --
  a sequence-to-image regression, which is what a ConvLSTM is for.
- Predicting a residual on top of the last frame means an untrained network
  is exactly the persistence baseline; training can only move away from it
  where the data supports doing so.
- Well under 1 M parameters: with ~20 training sequences from one 34-hour period, a
  large model would only memorise.

Importing this module requires torch. Nothing else in the imagery package
imports it at module load time -- `ml_inference.py` does so lazily, so the
backend still starts (and the classical pipeline still works) without torch.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_NAME = "Two-scale ConvLSTM encoder-decoder (residual on last frame, Laplace uncertainty head)"


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None):
        if state is None:
            zeros = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
            state = (zeros, zeros)
        hidden, cell = state
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, hidden], dim=1)), 4, dim=1)
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(cell)
        return hidden, cell


def _block(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    # Deliberately no normalisation layer: GroupNorm/InstanceNorm statistics
    # depend on what happens to be inside the crop, which would make a
    # 128-px training crop and a 448-px inference tile behave differently and
    # leave visible seams between tiles. Plain conv + SiLU is exactly
    # translation-equivariant, so tiled inference equals whole-image inference.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, dilation=dilation),
        nn.SiLU(inplace=True),
    )


class InsatNowcastNet(nn.Module):
    """Input  : (B, T, C, H, W) normalised brightness temperature, H and W
                 multiples of 4. Channel 0 must be the target channel (TIR1).
    Output : (mean, scale), each (B, n_leads, H, W), in the same normalised
             units. `mean` = last TIR1 frame + learned change; `scale` is the
             Laplace scale b (> 0), i.e. the predicted mean absolute error.

    All learned computation runs at 1/2 and 1/4 resolution (the input is
    space-to-depth'd, the output depth-to-space'd). Native-resolution detail
    is not lost, because the prediction is the full-resolution last frame
    plus a learned change -- and it is what makes training feasible on a
    laptop.
    """

    def __init__(self, in_channels: int = 2, base: int = 32, n_leads: int = 3, min_scale: float = 1e-3):
        super().__init__()
        self.in_channels, self.base, self.n_leads, self.min_scale = in_channels, base, n_leads, min_scale
        self.unshuffle = nn.PixelUnshuffle(2)
        self.stem = nn.Sequential(_block(4 * in_channels, base), _block(base, base))            # 1/2
        self.down = nn.Sequential(_block(base, 2 * base, stride=2), _block(2 * base, 2 * base))  # 1/4
        self.lstm1 = ConvLSTMCell(base, base, 3)
        self.lstm2 = ConvLSTMCell(2 * base, 2 * base, 3)
        # dilated convolutions widen the receptive field at 1/4 resolution so
        # that 90 minutes of cloud motion (tens of pixels) is in view.
        self.bottleneck = nn.Sequential(_block(2 * base, 2 * base, dilation=2), _block(2 * base, 2 * base, dilation=4))
        self.up = nn.Sequential(_block(4 * base, 2 * base), _block(2 * base, 2 * base))
        self.head = nn.Conv2d(2 * base, 2 * n_leads * 4, 1)
        self.shuffle = nn.PixelShuffle(2)
        # Start exactly at persistence: zero predicted change.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[4 * n_leads:] = -2.0  # softplus(-2) ~ 0.13 normalised units

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        _, steps, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")
        if height % 4 or width % 4:
            raise ValueError(f"H and W must be multiples of 4, got {height}x{width}")

        state1 = state2 = None
        half = None
        for t in range(steps):
            half = self.stem(self.unshuffle(x[:, t]))
            state1 = self.lstm1(half, state1)
            state2 = self.lstm2(self.down(half), state2)

        deep = self.bottleneck(state2[0])
        deep = F.interpolate(deep, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.shuffle(self.head(self.up(torch.cat([deep, state1[0], half], dim=1))))

        last = x[:, -1, 0:1]
        mean = last + out[:, : self.n_leads]
        scale = F.softplus(out[:, self.n_leads:]) + self.min_scale
        return mean, scale

    def config(self) -> dict:
        return {"in_channels": self.in_channels, "base": self.base, "n_leads": self.n_leads, "min_scale": self.min_scale}


@torch.no_grad()
def tiled_forward(model: "InsatNowcastNet", x, device: torch.device, tile: int = 448, halo: int = 48):
    """Run the model over an arbitrarily large (T, C, H, W) numpy array in
    overlapping tiles, keeping only each tile's interior (`halo` pixels of
    context are discarded on every side that is not a true image edge).
    Returns numpy (mean, scale), each (n_leads, H, W), normalised units.

    Bounded memory regardless of image size -- a full 1616x1737 sector does
    not fit comfortably in one pass on an 8 GB machine.
    """
    import numpy as np  # noqa: PLC0415

    _, _, height, width = x.shape
    mean = np.empty((model.n_leads, height, width), dtype=np.float32)
    scale = np.empty_like(mean)
    core = tile - 2 * halo
    for r in range(0, height, core):
        for c in range(0, width, core):
            r0, c0 = max(0, r - halo), max(0, c - halo)
            r1, c1 = min(height, r + core + halo), min(width, c + core + halo)
            patch = x[:, :, r0:r1, c0:c1]
            pad_r, pad_c = (-patch.shape[-2]) % 4, (-patch.shape[-1]) % 4
            if pad_r or pad_c:
                patch = np.pad(patch, ((0, 0), (0, 0), (0, pad_r), (0, pad_c)), mode="edge")
            m, s = model(torch.from_numpy(np.ascontiguousarray(patch))[None].to(device))
            rr, cc = min(height, r + core), min(width, c + core)
            mean[:, r:rr, c:cc] = m[0, :, r - r0:rr - r0, c - c0:cc - c0].float().cpu().numpy()
            scale[:, r:rr, c:cc] = s[0, :, r - r0:rr - r0, c - c0:cc - c0].float().cpu().numpy()
    return mean, scale


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nowcast_loss(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """L1 on the mean + Laplace negative log-likelihood for the scale, with
    the mean detached inside the likelihood term.

    Detaching keeps the two jobs separate: the mean is trained purely to be
    accurate (plain L1 -- no way to lower the loss by declaring a hard pixel
    'uncertain' and then ignoring it), and the scale is trained purely to
    report honestly how wrong that mean tends to be.

    `mask` is 1 where a target exists and should be scored (some lead times
    have no later frame; crop borders are excluded) and 0 elsewhere.
    """
    abs_err = (target - mean).abs()
    denom = mask.sum().clamp_min(1.0)
    l1 = (abs_err * mask).sum() / denom
    nll = ((abs_err.detach() / scale + torch.log(scale)) * mask).sum() / denom
    return l1 + nll, l1.detach(), nll.detach()
