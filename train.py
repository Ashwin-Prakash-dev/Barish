"""
Training script for TwinFloodNet — subfolder dataset layout.

Usage (Colab)
-------------
  Paste the cells below or run:
    !python train.py --data_dir "/content/drive/MyDrive/FloodDataset/SenForFlood/CEMS" \
                     --out_dir  "/content/drive/MyDrive/runs/exp1" \
                     --epochs 50 --batch_size 4 --amp

Split strategy
--------------
  By default the script holds out a fraction of EVENTS (not tiles) for
  validation, so the model is evaluated on unseen flood events.
  Use --val_events to name specific events explicitly.
"""

import argparse
import os
import time
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import SenForFloodsDataset, discover_samples
from model   import build_model
from losses  import DiceFocalLoss
from metrics import FloodMetrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str,   required=True,
                   help="Root folder, e.g. .../SenForFlood/CEMS")
    p.add_argument("--out_dir",     type=str,   default="runs")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--base_ch",     type=int,   default=32)
    p.add_argument("--no_aux",      action="store_true")
    p.add_argument("--val_split",   type=float, default=0.15,
                   help="Fraction of events held out for validation")
    p.add_argument("--val_events",  type=str,   nargs="+", default=None,
                   help="Explicit event names to use as val set, e.g. EMSN194 EMSN200")
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--amp",         action="store_true")
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_events(data_dir, val_split, val_events, seed):
    """Return (train_events, val_events) lists of event name strings."""
    samples     = discover_samples(data_dir)
    all_events  = sorted({en for _, _, en in samples})
    print(f"Found {len(all_events)} events: {all_events}")

    if val_events:
        val_set   = set(val_events)
        train_evs = [e for e in all_events if e not in val_set]
        val_evs   = [e for e in all_events if e in val_set]
    else:
        random.seed(seed)
        shuffled  = all_events[:]
        random.shuffle(shuffled)
        n_val     = max(1, int(len(shuffled) * val_split))
        val_evs   = shuffled[:n_val]
        train_evs = shuffled[n_val:]

    print(f"Train events ({len(train_evs)}): {train_evs}")
    print(f"Val   events ({len(val_evs)}):   {val_evs}")
    return train_evs, val_evs


def get_dataloaders(args):
    use_aux = not args.no_aux
    train_evs, val_evs = split_events(
        args.data_dir, args.val_split, args.val_events, args.seed
    )

    train_ds = SenForFloodsDataset(args.data_dir, use_aux=use_aux,
                                   augment=True,  events=train_evs)
    val_ds   = SenForFloodsDataset(args.data_dir, use_aux=use_aux,
                                   augment=False, events=val_evs)

    train_ds.summary()
    val_ds.summary()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, use_aux,
              scaler=None, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    metrics    = FloodMetrics()
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in loader:
            if use_aux:
                before, during, aux, target = [b.to(device) for b in batch]
            else:
                before, during, target = [b.to(device) for b in batch]
                aux = None

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(before, during, aux)
                loss   = criterion(logits, target)

            if training:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            metrics.update(logits.detach(), target)

    return total_loss / len(loader), metrics.compute()


def main():
    args   = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(args)

    use_aux = not args.no_aux
    model   = build_model(use_aux=use_aux, base_ch=args.base_ch).to(device)
    n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_p:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    criterion = DiceFocalLoss()
    scaler    = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    start_epoch, best_iou, history = 0, 0.0, []

    if args.resume and os.path.isfile(args.resume):
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("best_iou", 0.0)
        history     = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}, best IoU={best_iou:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer,
                                  device, use_aux, scaler, training=True)
        scheduler.step()
        vl_loss, vl_m = run_epoch(model, val_loader, criterion, optimizer,
                                  device, use_aux, training=False)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} "
            f"| tr_loss={tr_loss:.4f} tr_iou={tr_m['iou']:.4f} tr_f1={tr_m['f1']:.4f} "
            f"| vl_loss={vl_loss:.4f} vl_iou={vl_m['iou']:.4f} vl_f1={vl_m['f1']:.4f} "
            f"| {elapsed:.0f}s"
        )

        history.append({"epoch": epoch,
                        "tr_loss": tr_loss, "tr_iou": tr_m["iou"], "tr_f1": tr_m["f1"],
                        "vl_loss": vl_loss, "vl_iou": vl_m["iou"], "vl_f1": vl_m["f1"]})

        ckpt_data = {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_iou": best_iou, "history": history, "args": vars(args)}
        torch.save(ckpt_data, out_dir / "last.pth")

        if vl_m["iou"] > best_iou:
            best_iou = vl_m["iou"]
            ckpt_data["best_iou"] = best_iou
            torch.save(ckpt_data, out_dir / "best.pth")
            print(f"  -> New best IoU: {best_iou:.4f}  (saved best.pth)")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val IoU = {best_iou:.4f}")


if __name__ == "__main__":
    main()
