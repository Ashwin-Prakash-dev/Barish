"""
TwinFloodNet
============
A Siamese encoder-decoder that processes a (before, during) pair of
Sentinel-1 images and optionally fuses terrain + LULC auxiliary data
to produce a per-pixel flood probability map.

Architecture overview
---------------------
                ┌──────────────┐     ┌──────────────┐
  s1_before ──► │  Shared      │     │  Shared      │ ◄── s1_during
                │  Encoder     │     │  Encoder     │
                └──────┬───────┘     └──────┬───────┘
                       │  e_before          │  e_during
                       └────────┬───────────┘
                            change module
                         (diff + concat + conv)
                                 │
                    optional auxiliary injection
                         (terrain + LULC)
                                 │
                         Decoder (UNet-style)
                                 │
                         logit map (H × W × 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        pad = dilation * (kernel // 2)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Basic residual block (no bottleneck)."""
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnRelu(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class DownBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool, returns (after_pool, before_pool)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch, out_ch), ResBlock(out_ch))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat          # pooled, skip


class UpBlock(nn.Module):
    """Bilinear upsample → concat skip → Conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ResBlock(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd-sized feature maps
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (lightweight 4-branch)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.b0 = ConvBnRelu(in_ch, out_ch, 1)
        self.b1 = ConvBnRelu(in_ch, out_ch, dilation=6)
        self.b2 = ConvBnRelu(in_ch, out_ch, dilation=12)
        self.b3 = ConvBnRelu(in_ch, out_ch, dilation=18)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_ch, out_ch, 1),
        )
        self.proj = ConvBnRelu(out_ch * 5, out_ch, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), gap], dim=1))


class ChangeModule(nn.Module):
    """
    Fuse before/during feature pairs:
      difference + element-wise product + concatenation → Conv → ResBlock
    """
    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(feat_ch * 4, out_ch),   # diff, prod, f_b, f_d
            ResBlock(out_ch),
        )

    def forward(self, f_b, f_d):
        diff = f_b - f_d
        prod = f_b * f_d
        return self.conv(torch.cat([diff, prod, f_b, f_d], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Main network
# ─────────────────────────────────────────────────────────────────────────────

class TwinFloodNet(nn.Module):
    """
    Parameters
    ----------
    s1_in_ch    : number of SAR input channels per image (default 4)
    aux_in_ch   : auxiliary channels (terrain=2 + LULC one-hot=8 → 10); 0 = disabled
    base_ch     : base feature width; encoder uses [base, 2x, 4x, 8x]
    num_classes : 1 → binary flood segmentation (BCEWithLogitsLoss)
                  2 → multi-class (CrossEntropyLoss) – flood / no-flood
    """

    def __init__(
        self,
        s1_in_ch: int = 4,
        aux_in_ch: int = 10,
        base_ch: int = 32,
        num_classes: int = 1,
    ):
        super().__init__()
        c = base_ch
        self.num_classes = num_classes

        # ── Shared encoder (applied to both before and during) ──────────────
        self.enc1 = DownBlock(s1_in_ch, c)       # /2  → skip ch=c
        self.enc2 = DownBlock(c, c * 2)          # /4  → skip ch=2c
        self.enc3 = DownBlock(c * 2, c * 4)      # /8  → skip ch=4c
        self.enc4 = DownBlock(c * 4, c * 8)      # /16 → skip ch=8c

        # ── Bottleneck with ASPP ─────────────────────────────────────────────
        self.bottleneck = ASPP(c * 8, c * 8)     # operates on /16

        # ── Change modules at each scale ─────────────────────────────────────
        self.ch4 = ChangeModule(c * 8, c * 8)
        self.ch3 = ChangeModule(c * 4, c * 4)
        self.ch2 = ChangeModule(c * 2, c * 2)
        self.ch1 = ChangeModule(c,     c)

        # ── Optional auxiliary injection (into bottleneck) ───────────────────
        self.use_aux = aux_in_ch > 0
        if self.use_aux:
            self.aux_enc = nn.Sequential(
                ConvBnRelu(aux_in_ch, c * 2),
                nn.MaxPool2d(2, 2),
                ConvBnRelu(c * 2, c * 4),
                nn.MaxPool2d(2, 2),
                ConvBnRelu(c * 4, c * 8),
                nn.MaxPool2d(2, 2),
                ConvBnRelu(c * 8, c * 8),
                nn.MaxPool2d(2, 2),
            )
            self.aux_fuse = ConvBnRelu(c * 16, c * 8, 1)  # 2×c8 → c8

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec4 = UpBlock(c * 8, c * 4, c * 4)   # /16 → /8
        self.dec3 = UpBlock(c * 4, c * 2, c * 2)   # /8  → /4
        self.dec2 = UpBlock(c * 2, c,     c)        # /4  → /2
        self.dec1 = UpBlock(c,     c,     c)        # /2  → /1

        # ── Head ────────────────────────────────────────────────────────────
        self.head = nn.Conv2d(c, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(self, x):
        """Run shared encoder; returns (bottleneck, [skip1..skip4])."""
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        return x, [s1, s2, s3, s4]

    def forward(
        self,
        s1_before: torch.Tensor,
        s1_during: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        s1_before : (B, C, H, W)
        s1_during : (B, C, H, W)
        aux       : (B, A, H, W) or None

        Returns
        -------
        logits    : (B, num_classes, H, W)
        """
        # ── Encode both branches ──────────────────────────────────────────
        bot_b, skips_b = self._encode(s1_before)
        bot_d, skips_d = self._encode(s1_during)

        # ── Change at bottleneck ──────────────────────────────────────────
        bot = self.ch4(bot_b, bot_d)             # (B, 8c, H/16, W/16)

        # ── Optional auxiliary fusion ──────────────────────────────────────
        if self.use_aux and aux is not None:
            aux_feat = self.aux_enc(aux)          # (B, 8c, H/16, W/16)
            if aux_feat.shape[-2:] != bot.shape[-2:]:
                aux_feat = F.interpolate(
                    aux_feat, size=bot.shape[-2:], mode="bilinear", align_corners=False
                )
            bot = self.aux_fuse(torch.cat([bot, aux_feat], dim=1))

        # ── Change at skip connections ────────────────────────────────────
        cs4 = self.ch3(skips_b[2], skips_d[2])   # (B, 4c, H/8)
        cs3 = self.ch2(skips_b[1], skips_d[1])   # (B, 2c, H/4)
        cs2 = self.ch1(skips_b[0], skips_d[0])   # (B,  c, H/2)
        cs1 = (skips_b[0] + skips_d[0]) / 2      # (B, c, H/2) — averaged finest skip

        # ── Decode ────────────────────────────────────────────────────────
        x = self.dec4(bot, cs4)
        x = self.dec3(x,   cs3)
        x = self.dec2(x,   cs2)
        x = self.dec1(x,   cs1)

        return self.head(x)                       # (B, num_classes, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    use_aux: bool = True,
    num_classes: int = 1,
    base_ch: int = 32,
) -> TwinFloodNet:
    aux_ch = 10 if use_aux else 0   # 2 terrain + 8 LULC one-hot
    return TwinFloodNet(s1_in_ch=4, aux_in_ch=aux_ch, base_ch=base_ch, num_classes=num_classes)
