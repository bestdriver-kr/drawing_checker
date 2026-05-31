"""Simplified CRAFT-style text detector for engineering drawings.

Compared to the original Clova AI CRAFT (~20 M parameters, VGG16-based):
  * Smaller backbone (≈ 5–8 M parameters)
  * Same output structure: 2-channel heatmap (region + affinity), at 1/2
    input resolution
  * U-Net style encoder-decoder

The output stride is 2 — input is INPUT_SIZE×INPUT_SIZE, output is
HEATMAP_SIZE×HEATMAP_SIZE (defined in craft_data).  We use this stride
because the original CRAFT does too, and our heatmap label generation
already downsamples by 2.

The model is intentionally lean for the PoC.  If results are promising
we'd scale up to the full CRAFT for production.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int,
                  k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _Down(nn.Module):
    """Encoder block: 2 conv-bn-relu, then 2×2 max-pool downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feat = self.body(x)         # for skip
        return feat, self.pool(feat)


class _Up(nn.Module):
    """Decoder block: upsample, concat skip, conv-bn-relu."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.body = nn.Sequential(
            _conv_bn_relu(in_ch + skip_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad/crop just in case of off-by-one mismatches after odd strides
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        x = torch.cat([x, skip], dim=1)
        return self.body(x)


class CRAFT_Lite(nn.Module):
    """Lite CRAFT with output stride 2.

    Encoder downsamples: H → H/2 → H/4 → H/8 → H/16
    Decoder upsamples back to H/2 (stride-2 output) — matches the
    heatmap label resolution from `craft_data.HEATMAP_SIZE`.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        # Stem (no downsample yet): input H → H
        self.stem = nn.Sequential(
            _conv_bn_relu(3, base_ch),
            _conv_bn_relu(base_ch, base_ch),
        )
        # Encoder: H → H/2 → H/4 → H/8 → H/16
        self.down1 = _Down(base_ch,        base_ch * 2)   # H/2
        self.down2 = _Down(base_ch * 2,    base_ch * 4)   # H/4
        self.down3 = _Down(base_ch * 4,    base_ch * 8)   # H/8
        self.down4 = _Down(base_ch * 8,    base_ch * 8)   # H/16
        # Decoder back to H/2 (stride 2 output)
        self.up3 = _Up(base_ch * 8, base_ch * 8, base_ch * 4)  # H/8
        self.up2 = _Up(base_ch * 4, base_ch * 4, base_ch * 2)  # H/4
        self.up1 = _Up(base_ch * 2, base_ch * 2, base_ch)      # H/2

        # Output head: 2 channels (region, affinity)
        self.head = nn.Sequential(
            _conv_bn_relu(base_ch, base_ch),
            nn.Conv2d(base_ch, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W).  Returns (B, 2, H/2, W/2) with values in [0,1].
        Channel 0 = region heatmap, channel 1 = affinity heatmap."""
        s0 = self.stem(x)                  # H
        s1, x1 = self.down1(s0)            # s1: H,  x1: H/2
        s2, x2 = self.down2(x1)            # s2: H/2, x2: H/4
        s3, x3 = self.down3(x2)            # s3: H/4, x3: H/8
        _,  x4 = self.down4(x3)            #         x4: H/16

        y3 = self.up3(x4, s3)              # H/4
        y2 = self.up2(y3, s2)              # H/2
        y1 = self.up1(y2, s1)              # H

        # Final downsample to H/2 (stride 2 output)
        y = nn.functional.avg_pool2d(y1, 2)
        return self.head(y)


def build_model(device="cpu", base_ch: int = 32) -> CRAFT_Lite:
    model = CRAFT_Lite(base_ch=base_ch).to(device)
    return model
