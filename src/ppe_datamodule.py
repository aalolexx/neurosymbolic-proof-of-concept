# ppe_person_crop_dm.py (square crops that ENVELOPE the largest person bbox, fitted inside the image; no warping/padding)
from __future__ import annotations
import os, glob, random, math
from typing import List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms as T

CLASS_NAMES = [
    'Hardhat',        # 0
    'Mask',           # 1
    'NO-Hardhat',     # 2
    'NO-Mask',        # 3
    'NO-Safety Vest', # 4
    'Person',         # 5
    'Safety Cone',    # 6
    'Safety Vest',    # 7
    'machinery',      # 8
    'vehicle',        # 9
]
PERSON_CLASS_ID = 5

# -------- geometry helpers (PIL right/bottom are EXCLUSIVE) --------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _yolo_to_xyxy_pixels(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int,int,int,int]:
    """
    YOLO-normalized (cx,cy,w,h) -> pixel box (x1,y1,x2,y2) with right/bottom exclusive.
    floor(left/top), ceil(right/bottom) so the box is fully included.
    """
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

def _center_of_xyxy(x1:int,y1:int,x2:int,y2:int) -> Tuple[float,float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _point_inside_box(x:float,y:float, x1:int,y1:int,x2:int,y2:int) -> bool:
    # treat boundary as inside (x2,y2 are exclusive in crop math, but inclusive is fine for label logic)
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

def _square_envelope_box_fit_inside_image(
    x1:int, y1:int, x2:int, y2:int, margin: float, W:int, H:int
) -> Tuple[int,int,int,int]:
    """
    Make a SQUARE that (a) envelopes the person bbox expanded by `margin`,
    and (b) is fully INSIDE the image (no padding).
    Steps:
      1) Expand bbox by margin in its own width/height.
      2) Required square side = ceil(max(expanded_w, expanded_h)).
      3) Choose a center close to the expanded box center, then SHIFT so the square:
         - stays inside the image, and
         - still covers the expanded box.
    """
    bw = x2 - x1
    bh = y2 - y1
    # expanded (float, unclamped)
    ex1 = x1 - bw * margin
    ey1 = y1 - bh * margin
    ex2 = x2 + bw * margin
    ey2 = y2 + bh * margin

    # side required to cover expanded box (at least 1 px)
    req_side = max(ex2 - ex1, ey2 - ey1)
    side = int(math.ceil(max(1.0, req_side)))

    # If the needed square doesn't fit, cap to the largest possible side
    max_side = min(W, H)
    if side > max_side:
        side = max_side

    # desired center = expanded box center
    cx = (ex1 + ex2) / 2.0
    cy = (ey1 + ey2) / 2.0

    half = side / 2.0

    # To fully cover [ex1,ex2], center must satisfy:
    #   cx >= ex2 - half  and  cx <= ex1 + half
    # Combine with image-fit constraint:
    #   cx in [half, W - half]
    min_cx = max(ex2 - half,        half)
    max_cx = min(ex1 + half,  W - half)
    if min_cx > max_cx:
        # If impossible (happens when side < expanded width), just clamp to image.
        min_cx, max_cx = half, W - half
    cx = _clamp(cx, min_cx, max_cx)

    min_cy = max(ey2 - half,        half)
    max_cy = min(ey1 + half,  H - half)
    if min_cy > max_cy:
        min_cy, max_cy = half, H - half
    cy = _clamp(cy, min_cy, max_cy)

    # Build integer square box; keep right/bottom exclusive
    sx1 = int(math.floor(cx - half))
    sy1 = int(math.floor(cy - half))
    sx2 = sx1 + side
    sy2 = sy1 + side

    # Final clamp (in case of rounding)
    if sx1 < 0:
        sx2 -= sx1; sx1 = 0
    if sy1 < 0:
        sy2 -= sy1; sy1 = 0
    if sx2 > W:
        sx1 -= (sx2 - W); sx2 = W
    if sy2 > H:
        sy1 -= (sy2 - H); sy2 = H

    # Ensure at least 1px
    if sx2 <= sx1: sx2 = min(W, sx1 + 1)
    if sy2 <= sy1: sy2 = min(H, sy1 + 1)
    return sx1, sy1, sx2, sy2

# -------- dataset --------

class YoloPPEPersonCropDataset(Dataset):
    """
    Returns (square_person_crop, y[10]) where y[k]=1 iff any object of class k
    has its center inside the chosen PERSON bbox (class 5). y[5] is forced to 1.

    The returned image is a **square crop** that fully ENVELOPES the **largest**
    available person bbox (optionally expanded by `person_margin`) and is SHIFTED
    to fit inside the image. This guarantees: no padding artifacts and no aspect-ratio warping.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        person_margin: float = 0.05,
        include_person_always_one: bool = True,
        seed: Optional[int] = None,
        custom_transform: Optional[Any] = None,
        min_person_area_frac: float = 0.0,
        samples: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__()
        assert split in ("train", "valid", "test")
        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, split, "images")
        self.labels_dir = os.path.join(root_dir, split, "labels")
        self.img_size = img_size
        self.augment = augment
        self.person_margin = person_margin
        self.include_person_always_one = include_person_always_one
        self.rng = random.Random(seed)
        self.min_person_area_frac = max(0.0, float(min_person_area_frac))

        # Build or accept prefiltered samples
        if samples is not None:
            self.samples: List[Tuple[str,str]] = list(samples)
        else:
            self.samples = []
            label_paths = sorted(glob.glob(os.path.join(self.labels_dir, "*.txt")))
            for lp in label_paths:
                base = os.path.splitext(os.path.basename(lp))[0]
                ip = None
                # be robust to cases like .JPG, .PNG, .jpeg, etc.
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    cand = os.path.join(self.images_dir, base + ext)
                    if os.path.exists(cand):
                        ip = cand; break
                if ip is None:
                    continue

                recs = self._read_label_file(lp)
                if not recs:
                    continue
                try:
                    with Image.open(ip) as im:
                        W, H = im.size
                except Exception:
                    continue

                keep = False
                for (cid, cx, cy, w, h) in recs:
                    if cid != PERSON_CLASS_ID:
                        continue
                    x1,y1,x2,y2 = _yolo_to_xyxy_pixels(cx,cy,w,h,W,H)
                    bw = x2-x1; bh = y2-y1
                    if bw <= 0 or bh <= 0:
                        continue
                    if self.min_person_area_frac > 0.0:
                        if (bw*bh) / float(W*H) < self.min_person_area_frac:
                            continue
                    keep = True
                    break
                if keep:
                    self.samples.append((ip, lp))

        if not self.samples:
            raise RuntimeError(f"No images with a valid Person bbox in {self.split} split at {self.root_dir}")

        # transforms
        if custom_transform is not None:
            self.transform = custom_transform
        else:
            # crop is square -> square resize keeps aspect
            if augment:
                self.transform = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ])

    def _read_label_file(self, label_path: str) -> List[Tuple[int, float, float, float, float]]:
        out = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 5: continue
                try:
                    cid = int(float(parts[0]))
                    cx = float(parts[1]); cy = float(parts[2])
                    w  = float(parts[3]); h  = float(parts[4])
                    out.append((cid, cx, cy, w, h))
                except Exception:
                    continue
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        # Auto-orient using EXIF to keep boxes aligned, then ensure RGB
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W, H = img.size
        records = self._read_label_file(label_path)

        # collect person boxes (xyxy)
        person_boxes: List[Tuple[int,int,int,int]] = []
        for (cid, cx, cy, w, h) in records:
            if cid == PERSON_CLASS_ID:
                x1,y1,x2,y2 = _yolo_to_xyxy_pixels(cx,cy,w,h,W,H)
                # enforce min area if requested
                if self.min_person_area_frac > 0.0:
                    if ((x2-x1)*(y2-y1)) / (W*H) < self.min_person_area_frac:
                        continue
                person_boxes.append((x1,y1,x2,y2))
        if not person_boxes:
            raise IndexError("No valid person boxes at runtime; dataset may have changed.")

        # ---- CHANGE: choose the BIGGEST person bbox (by pixel area) instead of a random one
        px1, py1, px2, py2 = max(person_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

        # square crop that envelopes the person + margin, but stays inside the image
        sx1, sy1, sx2, sy2 = _square_envelope_box_fit_inside_image(
            px1, py1, px2, py2, self.person_margin, W, H
        )

        # build 10-dim labels using ORIGINAL person bbox
        y = [0]*len(CLASS_NAMES)
        for (cid, cx, cy, w, h) in records:
            x1,y1,x2,y2 = _yolo_to_xyxy_pixels(cx,cy,w,h,W,H)
            cxc, cyc = _center_of_xyxy(x1,y1,x2,y2)
            if _point_inside_box(cxc,cyc, px1,py1,px2,py2):
                y[cid] = 1
        if self.include_person_always_one:
            y[PERSON_CLASS_ID] = 1

        # crop + transform (square->square, no aspect distortion)
        crop = img.crop((sx1, sy1, sx2, sy2))
        crop_t = self.transform(crop)
        y_t = torch.tensor(y, dtype=torch.float32)
        return crop_t, y_t
