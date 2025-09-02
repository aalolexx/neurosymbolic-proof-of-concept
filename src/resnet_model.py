# ns_safe_module.py
from __future__ import annotations
import math
from typing import Optional, Tuple
import scallopy as scl

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lightning import LightningModule, Trainer, seed_everything
except ImportError:
    from pytorch_lightning import LightningModule, Trainer, seed_everything

from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


# ---- Lightning model ----
class SafeResnetModel(LightningModule):
    """
    ResNet18 backbone -> only one projection head -> is safe
    """
    def __init__(
        self,
        finetune_resnet: bool = True,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-1]))  # -> [B,512,1,1]        

        # Apply your desired semantics:
        if finetune_resnet:
            # Fine-tune only the last ~30% of the backbone
            _unfreeze_backbone_tail(self.feature_extractor, tail_fraction=0.30)
            _set_frozen_bn_eval(self.feature_extractor)
        else:
            # Freeze the whole backbone
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.embed_dim = 512

        # Projection heads -> neural predicates
        self.is_safe_out = make_predicate_head(self.embed_dim)
        self.is_safe_out.requires_grad_(True)
        # Losses
        self.bce = nn.BCELoss()

    # ---- utils ----
    def _extract_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> [B,512]
        feats = self.feature_extractor(x)          # [B,512,1,1]
        feats = feats.view(feats.size(0), -1)      # [B,512]
        return feats

    # ---- Lightning API ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only the final `person_is_safe` probability: shape [B]
        """
        feats = self._extract_embed(x)
        p_safe = torch.sigmoid(self.is_safe_out(feats)).squeeze(-1)
        return p_safe

    def _compute_targets(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dataset y is 10-d multi-hot: indices
        0=Helmet, 2=Safety Vest (we ignore NO-* for the core demo).
        """
        tgt_helmet = y[:, 0]  # Hardhat present inside person bbox
        tgt_vest = y[:, 2]  # Safety Vest present inside person bbox
        tgt_gloves = y[:, 1]  # Safety gloves present inside person bbox
        tgt_safe = (tgt_helmet * tgt_vest * tgt_gloves).clamp(0, 1)  # AND for supervision
        
        return tgt_helmet, tgt_vest, tgt_gloves, tgt_safe

     # -- shared step for train/val/test --
    def _shared_step(self, batch, stage: str):
        x, y = batch
        feats = self._extract_embed(x)
        feats = self._extract_embed(x)
        p_safe = torch.sigmoid(self.is_safe_out(feats)).squeeze(-1)

        tgt_helmet, tgt_vest, tgt_gloves, tgt_safe = self._compute_targets(y)

        loss_main = self.bce(p_safe, tgt_safe)
        #loss_aux  = self.hparams.aux_weight * (
        #    self.bce(p_helmet, tgt_helmet) + self.bce(p_vest, tgt_vest) + self.bce(p_gloves, tgt_gloves)
        #)
        loss = loss_main

        with torch.no_grad():
            acc = ( (p_safe >= 0.5).float() == tgt_safe ).float().mean()

        # Log settings
        on_step  = False
        on_epoch = True

        self.log_dict({
            f"{stage}/loss": loss,
            f"{stage}/loss_main": loss_main,
            f"{stage}/acc_safe": acc,
        }, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        if stage == "train":
            self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        elif stage == "val":
            self.log("val_loss",   loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    # -- Lightning hooks (DRY) --
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        _, _, _, p_safe = self.forward(x)
        return p_safe  # only person_is_safe

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ------------------------
# utils

def make_predicate_head(in_dim: int) -> nn.Sequential:
    """
    Two-hidden-layer MLP: in_dim -> 256 -> 128 -> 1 (logit).
    Uses ReLU and Kaiming init. Returns nn.Sequential.
    """
    mlp = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
    )
    # Kaiming init for ReLU MLP
    for m in mlp:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return mlp


def _unfreeze_backbone_tail(module: nn.Module, tail_fraction: float = 0.30):
    """
    Freeze entire backbone, then unfreeze the last `tail_fraction` of parameters.
    Works generically by parameter order (deepest layers appear last in ResNet18).
    """
    # freeze all
    for p in module.parameters():
        p.requires_grad = False

    # unfreeze the tail
    params = [p for p in module.parameters()]
    if not params:
        return
    k = max(1, int(round(len(params) * tail_fraction)))
    for p in params[-k:]:
        p.requires_grad = True

def _set_frozen_bn_eval(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            # If all params in this BN are frozen, keep it in eval
            if all((not p.requires_grad) for p in m.parameters(recurse=False)):
                m.eval()
