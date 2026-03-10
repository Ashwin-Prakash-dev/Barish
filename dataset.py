"""
SenForFloods Dataset Loader — nested subfolder layout
======================================================

Actual on-disk structure:
  <root>/
    CEMS/
      EMSN194/
        flood_mask/        000000_flood_mask.tif, 000001_flood_mask.tif, ...
        s1_before_flood/   000000_s1_before_flood.tif, ...
        s1_during_flood/   000000_s1_during_flood.tif, ...
        terrain/           000000_terrain.tif, ...
        LULC/              000000_LULC.tif, ...
      EMSN195/
        ...

Point root_dir at the folder that contains the CEMS folder (or directly
at a single event folder — the crawler handles both automatically).

Label convention:
  0   = no flood
  1   = flood
  255 = ignore  (permanent water class=2, or nodata)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────
# Per-channel normalisation (SAR converted to dB first)
# ──────────────────────────────────────────────────────────────────
S1_MEAN = np.array([-12.0, -19.0, -12.0, -19.0], dtype=np.float32)
S1_STD  = np.array([  5.0,   5.0,   5.0,   5.0], dtype=np.float32)

TERRAIN_MEAN = np.array([30.0, 5.0], dtype=np.float32)
TERRAIN_STD  = np.array([20.0, 5.0], dtype=np.float32)

LULC_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90]
LULC_NUM     = len(LULC_CLASSES)
LULC_MAP     = {v: i for i, v in enumerate(LULC_CLASSES)}


def safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 10.0 * np.log10(np.clip(x.astype(np.float32), eps, None))


def encode_lulc(lulc: np.ndarray) -> np.ndarray:
    out = np.zeros_like(lulc, dtype=np.int64)
    for raw, idx in LULC_MAP.items():
        out[lulc == raw] = idx
    return out


def _event_name_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for folder in ["flood_mask", "s1_before_flood", "s1_during_flood", "terrain", "LULC"]:
        if folder in parts:
            i = parts.index(folder)
            if i > 0:
                return parts[i - 1]
    return "unknown"


def _make_entry(tid, mask_dir, before_dir, during_dir, terrain_dir, lulc_dir):
    entry = {
        "tile_id": tid,
        "mask":    os.path.join(mask_dir,    f"{tid}_flood_mask.tif"),
        "before":  os.path.join(before_dir,  f"{tid}_s1_before_flood.tif"),
        "during":  os.path.join(during_dir,  f"{tid}_s1_during_flood.tif"),
        "terrain": os.path.join(terrain_dir, f"{tid}_terrain.tif"),
        "lulc":    os.path.join(lulc_dir,    f"{tid}_LULC.tif"),
    }
    for key in ("mask", "before", "during"):
        if not os.path.isfile(entry[key]):
            return None
    return entry


def _discover_samples(root_dir: str) -> List[dict]:
    samples = []

    # Layout A: flat files in root_dir
    flat_masks = sorted(glob.glob(os.path.join(root_dir, "*_flood_mask.tif")))
    for mask_path in flat_masks:
        tid = os.path.basename(mask_path).replace("_flood_mask.tif", "")
        d = os.path.dirname(mask_path)
        e = _make_entry(tid, d, d, d, d, d)
        if e:
            samples.append(e)
    if samples:
        return samples

    # Layout B: modality subfolders under event dirs
    for dirpath, _, _ in os.walk(root_dir):
        if os.path.basename(dirpath) != "flood_mask":
            continue
        ev = os.path.dirname(dirpath)
        for mask_path in sorted(glob.glob(os.path.join(dirpath, "*_flood_mask.tif"))):
            tid = os.path.basename(mask_path).replace("_flood_mask.tif", "")
            e = _make_entry(
                tid, dirpath,
                os.path.join(ev, "s1_before_flood"),
                os.path.join(ev, "s1_during_flood"),
                os.path.join(ev, "terrain"),
                os.path.join(ev, "LULC"),
            )
            if e:
                samples.append(e)
    return samples


class SenForFloodsDataset(Dataset):
    """
    Loads Sentinel-1 before/during flood tile pairs from the nested
    SenForFloods directory layout (multiple EMSN event folders).

    Parameters
    ----------
    root_dir   : top-level folder (the one containing CEMS/)
    use_aux    : include terrain + LULC auxiliary channels
    log_scale  : convert SAR linear amplitude to dB before normalisation
    augment    : random horizontal & vertical flips

    Returns per __getitem__
    -----------------------
    use_aux=True  ->  (before, during, aux, mask)   all torch.FloatTensor
    use_aux=False ->  (before, during, mask)
    """

    def __init__(self, root_dir, use_aux=True, log_scale=True, augment=False):
        self.use_aux   = use_aux
        self.log_scale = log_scale
        self.augment   = augment
        self.samples   = _discover_samples(root_dir)

        if not self.samples:
            raise FileNotFoundError(
                f"No tiles found under: {root_dir}\n"
                "Need subfolders: flood_mask/, s1_before_flood/, s1_during_flood/"
            )

        if use_aux:
            missing = sum(
                1 for s in self.samples
                if not os.path.isfile(s["terrain"]) or not os.path.isfile(s["lulc"])
            )
            if missing:
                print(f"[Dataset] WARNING: {missing} tiles missing terrain/LULC -> zeros used")

        events = {_event_name_from_path(s["mask"]) for s in self.samples}
        print(f"[Dataset] {len(self.samples)} tiles | {len(events)} event(s): {sorted(events)}")

    def _load_s1(self, path):
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if self.log_scale:
            img = safe_log(img)
        c = img.shape[2]
        img = (img - S1_MEAN[:c]) / S1_STD[:c]
        return img.transpose(2, 0, 1)

    def _load_terrain(self, path):
        img = tifffile.imread(path).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        c = img.shape[2]
        img = (img - TERRAIN_MEAN[:c]) / TERRAIN_STD[:c]
        return img.transpose(2, 0, 1)

    def _load_lulc(self, path):
        raw = tifffile.imread(path)
        idx = encode_lulc(raw)
        H, W = idx.shape
        ohe = np.zeros((LULC_NUM, H, W), dtype=np.float32)
        for c in range(LULC_NUM):
            ohe[c] = (idx == c).astype(np.float32)
        return ohe

    def _load_mask(self, path):
        mask = tifffile.imread(path).astype(np.int64)
        mask[mask == 2] = 255
        return mask

    def _augment(self, *arrays):
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-1).copy() for a in arrays)
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-2).copy() for a in arrays)
        return arrays

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        before = self._load_s1(s["before"])
        during = self._load_s1(s["during"])
        mask   = self._load_mask(s["mask"])
        H, W   = mask.shape

        if self.use_aux:
            terrain = (self._load_terrain(s["terrain"])
                       if os.path.isfile(s["terrain"])
                       else np.zeros((2, H, W), dtype=np.float32))
            lulc    = (self._load_lulc(s["lulc"])
                       if os.path.isfile(s["lulc"])
                       else np.zeros((LULC_NUM, H, W), dtype=np.float32))
            aux = np.concatenate([terrain, lulc], axis=0)
        else:
            aux = None

        if self.augment:
            to_aug = (before, during, aux, mask) if aux is not None else (before, during, mask)
            aug    = self._augment(*to_aug)
            if aux is not None:
                before, during, aux, mask = aug
            else:
                before, during, mask = aug

        before_t = torch.from_numpy(before.copy())
        during_t = torch.from_numpy(during.copy())
        mask_t   = torch.from_numpy(mask.copy())

        if aux is not None:
            return before_t, during_t, torch.from_numpy(aux.copy()), mask_t
        return before_t, during_t, mask_t

    def event_split(self, val_events: List[str]):
        """
        Split by event name to avoid geographic leakage between train/val.

        Usage:
            ds = SenForFloodsDataset("/content/drive/MyDrive/FloodDataset/SenForFlood")
            train_ds, val_ds = ds.event_split(["EMSN194", "EMSN210"])
        """
        train_s, val_s = [], []
        for s in self.samples:
            ev = _event_name_from_path(s["mask"])
            (val_s if ev in val_events else train_s).append(s)

        train_ds = _clone_dataset(self, train_s, augment=True)
        val_ds   = _clone_dataset(self, val_s,   augment=False)
        print(f"Split -> train: {len(train_ds)}  val: {len(val_ds)}  (val events: {val_events})")
        return train_ds, val_ds


def _clone_dataset(src, samples, augment):
    ds = object.__new__(SenForFloodsDataset)
    ds.use_aux   = src.use_aux
    ds.log_scale = src.log_scale
    ds.augment   = augment
    ds.samples   = samples
    return ds


# Public aliases for external scripts
discover_samples     = _discover_samples
event_name_from_path = _event_name_from_path
