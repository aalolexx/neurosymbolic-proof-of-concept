# train_and_checkpoint.py
from lightning import Trainer, seed_everything
try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except Exception:
    from pytorch_lightning.callbacks import ModelCheckpoint  # older versions

from neurosymbolic_model import SafeWithScallopLitModel
from ppe_dm_strict_prefilter import PPEDataModule

seed_everything(42)

# --- Data ---
dm = PPEDataModule(
    root_dir="../data/ppe/css-data",
    img_size=224,
    batch_size=64,
    num_workers=8,
    train_augment=True,
    person_margin=0.05,
    min_person_area_frac=0.0,
)

# --- Model ---
model = SafeWithScallopLitModel(
    lr=1e-3,
    backbone_trainable=True,
    aux_weight=0.2,
    provenance="diffprob",
)

# --- Checkpointing: save after EVERY epoch (plus a 'last.ckpt') ---
ckpt_cb = ModelCheckpoint(
    dirpath="../checkpoints/safe",   # folder to save into
    filename="safe-{epoch:02d}",  # file name template
    save_top_k=-1,                # keep ALL epochs
    every_n_epochs=1,             # save each epoch
    save_last=True,               # also keep 'last.ckpt'
    save_on_train_epoch_end=True  # ensure saving at train epoch end
)

# --- Trainer ---
trainer = Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    log_every_n_steps=25,
    callbacks=[ckpt_cb],
)

# --- Train ---
trainer.fit(model, datamodule=dm)

print("Last checkpoint path:", ckpt_cb.last_model_path)

# (Optional) Test best/last:
# trainer.test(model, datamodule=dm, ckpt_path=ckpt_cb.last_model_path)

# (Optional) Predict with a saved ckpt:
# loaded = SafeWithScallopLitModel.load_from_checkpoint(ckpt_cb.last_model_path)
# batch = next(iter(dm.val_dataloader()))
# p_safe = trainer.predict(loaded, dataloaders=[dm.val_dataloader()], ckpt_path=ckpt_cb.last_model_path)
