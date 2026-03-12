"""
TwinFloodNet
============
A Siamese encoder-decoder that processes a (before, during) pair of
Sentinel-1 images and optionally fuses terrain + LULC auxiliary data
to produce a per-pixel flood probability map.

Encoder
-------
Uses a pretrained backbone from segmentation-models-pytorch (default: resnet34).
Both branches share the same encoder weights (Siamese).
The first conv is automatically adapted by smp for 4-channel SAR input by
averaging the pretrained RGB weights across the channel dimension.

Architecture overview
---------------------
  s1_before ──► [ Shared pretrained encoder ] ──► feats_b [/2 … /32]
  s1_during ──► [ Shared pretrained encoder ] ──► feats_d [/2 … /32]
                            │
              ChangeModule + CBAM at each scale
                            │
              optional auxiliary injection (terrain + LULC)
                            │
              Decoder UpBlocks (with Dropout2d)
                            │
              Main logit map + deep supervision heads (training only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


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


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module — channel + spatial gates.
    Applied after each ChangeModule to focus on flood-relevant change patterns.
    """
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(1, ch // reduction)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.channel_att(x).unsqueeze(-1).unsqueeze(-1)
        x = x * w
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        x   = x * self.spatial_att(torch.cat([avg, mx], dim=1))
        return x


class UpBlock(nn.Module):
    """Bilinear upsample → concat skip → Conv → Dropout2d."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ResBlock(out_ch),
            nn.Dropout2d(dropout),
        )

    def forward(self, x, skip):
        x = self.up(x)
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
      difference + element-wise product + concatenation → Conv → ResBlock → CBAM
    """
    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(feat_ch * 4, out_ch),
            ResBlock(out_ch),
        )
        self.attn = CBAM(out_ch)

    def forward(self, f_b, f_d):
        diff = f_b - f_d
        prod = f_b * f_d
        x = self.conv(torch.cat([diff, prod, f_b, f_d], dim=1))
        return self.attn(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main network
# ─────────────────────────────────────────────────────────────────────────────

class TwinFloodNet(nn.Module):
    """
    Parameters
    ----------
    s1_in_ch         : SAR input channels per image (default 4)
    aux_in_ch        : auxiliary channels (terrain=2 + LULC one-hot=8 → 10); 0 = disabled
    base_ch          : controls decoder output widths (decoders output max(base_ch, enc_ch//2))
    num_classes      : 1 → binary segmentation
    dropout          : Dropout2d probability in all UpBlocks (default 0.2)
    encoder_name     : any smp-supported encoder, e.g. 'resnet34', 'efficientnet-b0'
    encoder_weights  : 'imagenet' (default) or None (random init)

    Encoder output channels (resnet34, depth=5, in_ch=4):
        (4, 64, 64, 128, 256, 512)
         ↑   ↑   ↑   ↑    ↑    ↑
        /1  /2  /4  /8  /16  /32
        input  skip features   bottleneck

    The architecture auto-configures ChangeModules and decoder dimensions
    from encoder.out_channels, so swapping the encoder name just works.

    Deep supervision
    ----------------
    During training returns (main_logits, ds3_logits, ds2_logits).
    During eval returns only main_logits — predict.py is unaffected.
    """

    def __init__(
        self,
        s1_in_ch: int = 4,
        aux_in_ch: int = 10,
        base_ch: int = 32,
        num_classes: int = 1,
        dropout: float = 0.2,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.num_classes = num_classes

        # ── Pretrained encoder (shared Siamese branch) ───────────────────────
        # smp automatically adapts the first conv for in_channels != 3
        # by averaging pretrained RGB weights across channel dim then tiling.
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=s1_in_ch,
            depth=5,
            weights=encoder_weights,
        )
        enc_chs  = self.encoder.out_channels   # e.g. (4, 64, 64, 128, 256, 512)
        skip_chs = enc_chs[1:-1]               # per-scale skip channels: finest → coarsest
        bot_ch   = enc_chs[-1]                 # bottleneck channel count
        n_skips  = len(skip_chs)               # 4 for depth=5

        # ── ASPP on bottleneck ───────────────────────────────────────────────
        self.bottleneck = ASPP(bot_ch, bot_ch)

        # ── ChangeModule at bottleneck and each skip scale ───────────────────
        # change_bottleneck: applied to the deepest encoder features before ASPP
        self.change_bottleneck = ChangeModule(bot_ch, bot_ch)
        # change_modules[0]=finest skip … change_modules[-1]=coarsest skip
        self.change_modules = nn.ModuleList([
            ChangeModule(ch, ch) for ch in skip_chs
        ])

        # ── Optional auxiliary injection (into bottleneck) ───────────────────
        self.use_aux = aux_in_ch > 0
        if self.use_aux:
            c = base_ch
            self.aux_enc = nn.Sequential(
                ConvBnRelu(aux_in_ch, c * 2),
                nn.MaxPool2d(2, 2),
                ConvBnRelu(c * 2, c * 4),
                nn.MaxPool2d(2, 2),
                ConvBnRelu(c * 4, c * 8),
                nn.MaxPool2d(2, 2),
                ConvBnRelu(c * 8, bot_ch),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),   # 5 total pools → /32, matches encoder depth=5
            )
            self.aux_fuse = ConvBnRelu(bot_ch * 2, bot_ch, 1)

        # ── Decoder — auto-configured from encoder channels ──────────────────
        # Skip order for decoder: coarsest → finest  (reversed skip_chs)
        rev_skip_chs = list(reversed(skip_chs))          # e.g. [256, 128, 64, 64]
        dec_out_chs  = [max(base_ch, ch // 2) for ch in rev_skip_chs]  # [128, 64, 32, 32]

        self.dec_blocks = nn.ModuleList()
        in_ch = bot_ch
        for skip_ch, out_ch in zip(rev_skip_chs, dec_out_chs):
            self.dec_blocks.append(UpBlock(in_ch, skip_ch, out_ch, dropout=dropout))
            in_ch = out_ch

        # Final UpBlock: /2 → /1, uses raw averaged finest skip
        self.dec_final = UpBlock(in_ch, skip_chs[0], base_ch, dropout=dropout)

        # ── Main head ────────────────────────────────────────────────────────
        self.head = nn.Conv2d(base_ch, num_classes, 1)

        # ── Deep supervision heads (training only) ───────────────────────────
        # ds_head3: after dec_blocks[1] at /8   (dec_out_chs[1])
        # ds_head2: after dec_blocks[2] at /4   (dec_out_chs[2])
        self.ds_head3 = nn.Conv2d(dec_out_chs[1], num_classes, 1)
        self.ds_head2 = nn.Conv2d(dec_out_chs[2], num_classes, 1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Initialise only decoder/head weights; encoder keeps pretrained weights."""
        scratch_modules = [
            self.bottleneck, self.change_bottleneck, self.change_modules,
            self.dec_blocks, self.dec_final, self.head, self.ds_head3, self.ds_head2,
        ]
        if self.use_aux:
            scratch_modules += [self.aux_enc, self.aux_fuse]
        for module in scratch_modules:
            for m in module.modules() if hasattr(module, "modules") else [module]:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def encoder_parameters(self):
        """Encoder params — trained at lower LR to preserve pretrained weights."""
        return list(self.encoder.parameters())

    def decoder_parameters(self):
        """All non-encoder params — trained at full LR."""
        enc_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

    def forward(
        self,
        s1_before: torch.Tensor,
        s1_during: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Parameters
        ----------
        s1_before : (B, C, H, W)
        s1_during : (B, C, H, W)
        aux       : (B, A, H, W) or None

        Returns
        -------
        eval  : main_logits  (B, num_classes, H, W)
        train : (main_logits, ds3_logits, ds2_logits)
        """
        # ── Encode both branches (shared weights) ─────────────────────────
        feats_b = self.encoder(s1_before)   # list of 6: [input, /2, /4, /8, /16, /32]
        feats_d = self.encoder(s1_during)

        # ── Change + ASPP at bottleneck (/32) ─────────────────────────────
        bot = self.change_bottleneck(feats_b[-1], feats_d[-1])
        bot = self.bottleneck(bot)

        # ── Optional auxiliary fusion ──────────────────────────────────────
        if self.use_aux and aux is not None:
            aux_feat = self.aux_enc(aux)
            if aux_feat.shape[-2:] != bot.shape[-2:]:
                aux_feat = F.interpolate(
                    aux_feat, size=bot.shape[-2:], mode="bilinear", align_corners=False
                )
            bot = self.aux_fuse(torch.cat([bot, aux_feat], dim=1))

        # ── ChangeModule at each skip scale ───────────────────────────────
        # feats[1:-1] = [/2, /4, /8, /16]; change_modules[i] matches feats[i+1]
        change_skips = [
            self.change_modules[i](feats_b[i + 1], feats_d[i + 1])
            for i in range(len(self.change_modules))
        ]
        # change_skips[0]=finest(/2) … change_skips[-1]=coarsest(/16)

        # Raw averaged finest skip for dec_final (/2)
        avg_finest = (feats_b[1] + feats_d[1]) / 2

        # ── Decode coarsest → finest ───────────────────────────────────────
        x = bot
        dec_outputs = []
        for i, dec_block in enumerate(self.dec_blocks):
            skip = change_skips[-(i + 1)]   # coarsest first
            x = dec_block(x, skip)
            dec_outputs.append(x)

        x    = self.dec_final(x, avg_finest)
        # smp stem outputs at /2 (unlike the old scratch DownBlock which kept /1),
        # so dec_final lands at /2 — one final 2x upsample restores full resolution.
        x    = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        main = self.head(x)

        if self.training:
            h, w = main.shape[-2:]
            ds3 = F.interpolate(
                self.ds_head3(dec_outputs[1]), size=(h, w), mode="bilinear", align_corners=False
            )
            ds2 = F.interpolate(
                self.ds_head2(dec_outputs[2]), size=(h, w), mode="bilinear", align_corners=False
            )
            return main, ds3, ds2

        return main


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    use_aux: bool = True,
    num_classes: int = 1,
    base_ch: int = 32,
    dropout: float = 0.2,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> TwinFloodNet:
    aux_ch = 10 if use_aux else 0
    return TwinFloodNet(
        s1_in_ch=4,
        aux_in_ch=aux_ch,
        base_ch=base_ch,
        num_classes=num_classes,
        dropout=dropout,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )
