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

# ---- Optional Scallop integration (fallbacks to t-norm if not available) ----
class ScallopAnd:
    """
    Minimal wrapper around a 2-predicate Scallop program:
        person_is_safe() :- helmet(), vest().
    If scallopy is not installed, falls back to product t-norm: p_safe = p_h * p_v
    """
    def __init__(self, provenance: str = "diffprob"):
        self.ok = False
        self.ctx = None
        
        self.scl = scl
        # Build a tiny program once; we’ll feed facts each forward pass.
        self.ctx = scl.ScallopContext(provenance=provenance)

        # Declare predicates
        self.ctx.add_relation("helmet", [("unit", "unit")])
        self.ctx.add_relation("vest", [("unit", "unit")])
        self.ctx.add_relation("person_is_safe", [("unit", "unit")])

        # person_is_safe() :- helmet(), vest().
        self.ctx.add_rule("person_is_safe(unit()) :- helmet(unit()), vest(unit()).")

    def __call__(self, p_helmet: torch.Tensor, p_vest: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_helmet: (B,) probabilities in [0,1]
            p_vest:   (B,) probabilities in [0,1]
        Returns:
            p_safe:   (B,) probabilities in [0,1]
        """

        # With scallopy, we’ll push one mini “database” per batch element.
        # This is a simple (and not the most efficient) reference implementation.
        p_safe = []
        for ph, pv in zip(p_helmet.detach().cpu().tolist(), p_vest.detach().cpu().tolist()):
            # New runtime for each item
            rt = self.ctx.run()  # snapshot the compiled program
            # Add weighted facts (fuzzy facts). We encode with (probability, tuple)
            # Some scallopy builds use rt.add_fact("pred", prob, args...) or rt.add_facts list.
            try:
                rt.add_fact("helmet", ph, ())
                rt.add_fact("vest", pv, ())
            except Exception:
                # API variant
                # TODO Check if this can be removed
                rt.add_fact("helmet", (ph, ()))
                rt.add_fact("vest", (pv, ()))

            out = rt.run("person_is_safe")  # returns list of (prob, args)
            if len(out) == 0:
                p_safe.append(0.0)
            else:
                # Aggregate (usually there’s one result)
                prob = 0.0
                for item in out:
                    # item may be (p, args) or (args, p) depending on version
                    if isinstance(item[0], (float, int)):
                        prob = max(prob, float(item[0]))
                    else:
                        prob = max(prob, float(item[-1]))
                p_safe.append(prob)
        return torch.tensor(p_safe, device=p_helmet.device, dtype=p_helmet.dtype)


# ---- Lightning model ----
class SafeWithScallopLitModel(LightningModule):
    """
    ResNet18 backbone -> two projection heads -> Scallop AND -> person_is_safe prob.
    Trains only on the final `person_is_safe` target (optionally with small aux losses).
    """
    def __init__(
        self,
        backbone_trainable: bool = True,
        provenance: str = "diffprob",
        aux_weight: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if not backbone_trainable:
            for p in resnet.parameters():
                p.requires_grad = False

        # Remove classifier head; keep convs + global avgpool
        # ResNet18: last conv output 512 channels + AdaptiveAvgPool2d
        self.feature_extractor = nn.Sequential(
            *(list(resnet.children())[:-1])  # outputs [B, 512, 1, 1]
        )
        self.embed_dim = 512

        # Projection heads -> neural predicates
        self.helmet_head = nn.Linear(self.embed_dim, 1)
        self.vest_head   = nn.Linear(self.embed_dim, 1)

        # Scallop rule (or differentiable fallback)
        self.scallop_program = ScallopAnd(provenance=provenance)

        # Losses
        self.bce = nn.BCELoss()

    # ---- utils ----
    def _extract_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> [B,512]
        feats = self.feature_extractor(x)          # [B,512,1,1]
        feats = feats.view(feats.size(0), -1)      # [B,512]
        return feats

    def _predicate_probs(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sigmoid to get probabilities
        p_helmet = torch.sigmoid(self.helmet_head(feats)).squeeze(-1)  # [B]
        p_vest   = torch.sigmoid(self.vest_head(feats)).squeeze(-1)    # [B]
        return p_helmet, p_vest

    def _scallop_prob(self, p_helmet: torch.Tensor, p_vest: torch.Tensor) -> torch.Tensor:
        # Through Scallop program (or t-norm fallback)
        return self.scallop_program(p_helmet, p_vest)

    # ---- Lightning API ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only the final `person_is_safe` probability: shape [B]
        """
        feats = self._extract_embed(x)
        p_helmet, p_vest = self._predicate_probs(feats)
        p_safe = self._scallop_prob(p_helmet, p_vest)
        return p_safe

    def _compute_targets(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dataset y is 10-d multi-hot: indices
        0=Hardhat, 7=Safety Vest (we ignore NO-* for the core demo).
        """
        tgt_helmet = y[:, 0]  # Hardhat present inside person bbox
        tgt_vest   = y[:, 7]  # Safety Vest present inside person bbox
        tgt_safe   = (tgt_helmet * tgt_vest).clamp(0, 1)  # AND for supervision
        return tgt_helmet, tgt_vest, tgt_safe

     # -- shared step for train/val/test --
    def _shared_step(self, batch, stage: str):
        x, y = batch
        feats = self._extract_embed(x)
        p_helmet, p_vest = self._predicate_probs(feats)
        p_safe = self._scallop_prob(p_helmet, p_vest)

        tgt_helmet, tgt_vest, tgt_safe = self._compute_targets(y)

        loss_main = self.bce(p_safe, tgt_safe)
        loss_aux  = self.hparams.aux_weight * (
            self.bce(p_helmet, tgt_helmet) + self.bce(p_vest, tgt_vest)
        )
        loss = loss_main + loss_aux

        with torch.no_grad():
            acc = ( (p_safe >= 0.5).float() == tgt_safe ).float().mean()

        # Log settings
        on_step  = (stage == "train")
        on_epoch = True
        self.log_dict({
            f"{stage}/loss": loss,
            f"{stage}/loss_main": loss_main,
            f"{stage}/loss_aux": loss_aux,
            f"{stage}/acc_safe": acc,
        }, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

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
        p_safe = self.forward(x)
        return p_safe  # only person_is_safe

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
