"""
Loss functions for flood segmentation — AMP safe.

All losses work directly on raw logits (no sigmoid before passing in).
They handle the ignore index (255) internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def _flatten_logits(pred: torch.Tensor, target: torch.Tensor):
    """
    Returns (logits, targets) as 1-D float tensors, ignore pixels removed.
    Works on raw logits — no sigmoid applied here.
    """
    pred   = pred.squeeze(1).float()           # (B,H,W)
    valid  = target != IGNORE_INDEX
    return pred[valid], target[valid].float()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        logits, tgt = _flatten_logits(pred, target)
        prob  = torch.sigmoid(logits)          # sigmoid inside, float32
        inter = (prob * tgt).sum()
        union = prob.sum() + tgt.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        logits, tgt = _flatten_logits(pred, target)
        # Use BCE-with-logits (AMP safe), then apply focal weighting
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        prob = torch.sigmoid(logits)
        pt   = torch.where(tgt == 1, prob, 1 - prob)
        w    = torch.where(tgt == 1,
                           torch.full_like(pt, self.alpha),
                           torch.full_like(pt, 1 - self.alpha))
        return (w * (1 - pt) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred, target):
        logits, tgt = _flatten_logits(pred, target)
        prob = torch.sigmoid(logits)
        tp = (prob * tgt).sum()
        fp = (prob * (1 - tgt)).sum()
        fn = ((1 - prob) * tgt).sum()
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


class DiceFocalLoss(nn.Module):
    """Recommended default: equal mix of Dice + Focal. Fully AMP safe."""
    def __init__(self, dice_w=0.5, focal_w=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.dice_w  = dice_w
        self.focal_w = focal_w
        self.dice    = DiceLoss()
        self.focal   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        return self.dice_w * self.dice(pred, target) + self.focal_w * self.focal(pred, target)


class ComboLoss(nn.Module):
    """Weighted BCE-with-logits + Dice."""
    def __init__(self, bce_w=0.5, dice_w=0.5, pos_weight=5.0):
        super().__init__()
        self.bce_w  = bce_w
        self.dice_w = dice_w
        self.dice   = DiceLoss()
        self.pos_w  = pos_weight

    def forward(self, pred, target):
        logits, tgt = _flatten_logits(pred, target)
        pos_weight  = torch.tensor([self.pos_w], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight)
        return self.bce_w * bce + self.dice_w * self.dice(pred, target)
