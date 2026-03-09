"""
Segmentation metrics for flood detection.

All metrics operate on binary tensors and skip IGNORE_INDEX (255) pixels.
"""

import torch
import numpy as np

IGNORE_INDEX = 255


class FloodMetrics:
    """
    Accumulates predictions across an epoch; call .compute() at the end.

    Usage
    -----
        metrics = FloodMetrics(threshold=0.5)
        for pred, target in loader:
            metrics.update(pred, target)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred   : (B, 1, H, W) raw logits
        target : (B, H, W)    int64, 255=ignore
        """
        prob  = torch.sigmoid(pred.squeeze(1))    # (B,H,W)
        binary = (prob >= self.threshold).long()

        valid  = target != IGNORE_INDEX
        b = binary[valid]
        t = target[valid]

        self.tp += int(((b == 1) & (t == 1)).sum())
        self.fp += int(((b == 1) & (t == 0)).sum())
        self.fn += int(((b == 0) & (t == 1)).sum())
        self.tn += int(((b == 0) & (t == 0)).sum())

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-7

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = tp / (tp + fp + fn + eps)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)

        return {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "iou":       iou,
            "accuracy":  accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def __repr__(self):
        r = self.compute()
        return (
            f"FloodMetrics | "
            f"F1={r['f1']:.4f}  IoU={r['iou']:.4f}  "
            f"P={r['precision']:.4f}  R={r['recall']:.4f}  "
            f"Acc={r['accuracy']:.4f}"
        )
