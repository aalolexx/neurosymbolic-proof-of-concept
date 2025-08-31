# ppe_dm_strict_prefilter.py
from __future__ import annotations
import os, glob, math
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader

try:
    from lightning import LightningDataModule
except ImportError:
    from pytorch_lightning import LightningDataModule

# import the dataset class you already have (with 'samples' param support)
from ppe_datamodule import YoloPPEPersonCropDataset, PERSON_CLASS_ID

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _yolo_to_xyxy_pixels(cx: float, cy: float, w: float, h: float, W: int, H: int):
    """YOLO-normalized -> pixel (x1,y1,x2,y2) with right/bottom exclusive."""
    cx = _clamp(cx, 0.0, 1.0); cy = _clamp(cy, 0.0, 1.0)
    w  = max(0.0, min(1.0, w)); h  = max(0.0, min(1.0, h))
    x1f = (cx - w/2.0) * W
    y1f = (cy - h/2.0) * H
    x2f = (cx + w/2.0) * W
    y2f = (cy + h/2.0) * H
    x1f = _clamp(x1f, 0.0, float(W))
    y1f = _clamp(y1f, 0.0, float(H))
    x2f = _clamp(x2f, 0.0, float(W))
    y2f = _clamp(y2f, 0.0, float(H))
    x1 = int(math.floor(x1f))
    y1 = int(math.floor(y1f))
    x2 = int(math.ceil(x2f))
    y2 = int(math.ceil(y2f))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def _read_label_file(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Reads YOLO .txt; returns list of (class, cx, cy, w, h). Skips malformed lines."""
    out = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx = float(parts[1]); cy = float(parts[2])
                w  = float(parts[3]); h  = float(parts[4])
                out.append((cid, cx, cy, w, h))
            except Exception:
                continue
    return out

class PPEDataModule(LightningDataModule):
    """
    Strict pre-filtering:
      - During setup, we scan each split's label files, map to an existing image,
        open the image to get W,H, and keep ONLY those samples with at least one
        valid Person (class 5) bbox of positive pixel area.
      - These filtered sample lists are passed into YoloPPEPersonCropDataset,
        so non-person images are never loaded into the datasets.
    """
    def __init__(
        self,
        root_dir: str,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        train_augment: bool = True,
        seed: Optional[int] = 42,
        person_margin: float = 0.05,
        min_person_area_frac: float = 0.0,   # e.g., 0.002 to drop tiny people up-front
        verbose: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_augment = train_augment
        self.seed = seed
        self.person_margin = person_margin
        self.min_person_area_frac = max(0.0, float(min_person_area_frac))
        self.verbose = verbose

        self.train_samples: Optional[List[Tuple[str,str]]] = None
        self.val_samples: Optional[List[Tuple[str,str]]] = None
        self.test_samples: Optional[List[Tuple[str,str]]] = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _find_image_for_label(self, images_dir: str, base: str) -> Optional[str]:
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(images_dir, base + ext)
            if os.path.exists(cand):
                return cand
        return None

    def _prefilter_split(self, split: str) -> List[Tuple[str, str]]:
        images_dir = os.path.join(self.root_dir, split, "images")
        labels_dir = os.path.join(self.root_dir, split, "labels")
        label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
        kept: List[Tuple[str,str]] = []

        for lp in label_paths:
            base = os.path.splitext(os.path.basename(lp))[0]
            ip = self._find_image_for_label(images_dir, base)
            if ip is None:
                continue

            # read labels and image size
            recs = _read_label_file(lp)
            if not recs:
                continue

            try:
                with Image.open(ip) as im:
                    W, H = im.size
            except Exception:
                continue

            # keep if there's at least one VALID person bbox
            keep = False
            for (cid, cx, cy, w, h) in recs:
                if cid != PERSON_CLASS_ID:
                    continue
                x1,y1,x2,y2 = _yolo_to_xyxy_pixels(cx, cy, w, h, W, H)
                bw = x2 - x1; bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                if self.min_person_area_frac > 0.0:
                    if (bw * bh) / float(W * H) < self.min_person_area_frac:
                        continue
                keep = True
                break

            if keep:
                kept.append((ip, lp))

        if self.verbose:
            print(f"[PPEDataModule] Split '{split}': kept {len(kept)} samples with a valid Person bbox.")

        if not kept:
            raise RuntimeError(f"No valid person-containing samples in split '{split}'. "
                               f"Check your dataset path or class ids.")
        return kept

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_samples = self._prefilter_split("train")
            self.val_samples   = self._prefilter_split("valid")

            self.train_ds = YoloPPEPersonCropDataset(
                root_dir=self.root_dir,
                split="train",
                img_size=self.img_size,
                augment=self.train_augment,
                seed=self.seed,
                person_margin=self.person_margin,
                min_person_area_frac=self.min_person_area_frac,
                # hand over the pre-filtered sample list
                samples=self.train_samples,
            )
            self.val_ds = YoloPPEPersonCropDataset(
                root_dir=self.root_dir,
                split="valid",
                img_size=self.img_size,
                augment=False,
                seed=self.seed,
                person_margin=self.person_margin,
                min_person_area_frac=self.min_person_area_frac,
                samples=self.val_samples,
            )

        if stage in (None, "test"):
            self.test_samples = self._prefilter_split("test")
            self.test_ds = YoloPPEPersonCropDataset(
                root_dir=self.root_dir,
                split="test",
                img_size=self.img_size,
                augment=False,
                seed=self.seed,
                person_margin=self.person_margin,
                min_person_area_frac=self.min_person_area_frac,
                samples=self.test_samples,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
