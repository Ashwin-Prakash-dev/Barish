"""
Inference script for TwinFloodNet.

Runs prediction on a directory of tile pairs and writes:
  - per-tile flood probability maps (.npy)
  - binary flood masks (.tif, 8-bit)
  - optional visualisations (.png)

Usage
-----
  python predict.py --checkpoint runs/best.pth \
                    --data_dir /path/to/test_tiles \
                    --out_dir predictions/ \
                    --visualise
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import tifffile

from dataset import SenForFloodsDataset
from model   import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--out_dir",    default="predictions")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--no_aux",     action="store_true")
    p.add_argument("--visualise",  action="store_true", help="Save RGB composite + overlay PNGs")
    return p.parse_args()


def visualise(before_np, during_np, prob_map, mask_gt, sid, out_dir):
    """Save a 4-panel matplotlib figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["S1 Before (VV)", "S1 During (VV)", "Flood Prob", "GT Mask"]
    imgs   = [
        before_np[0],                    # VV channel
        during_np[0],
        prob_map,
        mask_gt.astype(float),
    ]
    cmaps = ["gray", "gray", "RdYlGn_r", "coolwarm"]

    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0 if "Prob" in title or "Mask" in title else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sid}_vis.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    use_aux = not (args.no_aux or not saved_args.get("use_aux", True))
    base_ch = saved_args.get("base_ch", 32)

    model = build_model(use_aux=use_aux, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']+1}, best IoU={ckpt.get('best_iou', '?')})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = SenForFloodsDataset(args.data_dir, use_aux=use_aux, augment=False)
    print(f"Predicting {len(ds)} tiles …")

    # ── Inference loop ────────────────────────────────────────────────────────
    with torch.no_grad():
        for idx in range(len(ds)):
            sid = ds.samples[idx]["tile_id"]
            batch = ds[idx]

            if use_aux:
                before, during, aux, target = batch
                aux = aux.unsqueeze(0).to(device)
            else:
                before, during, target = batch
                aux = None

            before = before.unsqueeze(0).to(device)
            during = during.unsqueeze(0).to(device)

            logits   = model(before, during, aux)            # (1,1,H,W)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)
            binary   = (prob_map >= args.threshold).astype(np.uint8)

            # Save artefacts
            np.save(out_dir / f"{sid}_prob.npy",    prob_map)
            tifffile.imwrite(out_dir / f"{sid}_pred_mask.tif", binary)

            if args.visualise:
                mask_gt = target.numpy()
                before_np = before.squeeze(0).cpu().numpy()
                during_np = during.squeeze(0).cpu().numpy()
                visualise(before_np, during_np, prob_map, mask_gt, sid, str(out_dir))

            print(f"  [{idx+1}/{len(ds)}] {sid}  flood_frac={binary.mean():.3f}")

    print(f"\nPredictions saved to {out_dir}")


if __name__ == "__main__":
    main()
