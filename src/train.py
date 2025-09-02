# train_and_checkpoint.py
from pytorch_lightning import Trainer, seed_everything
try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except Exception:
    from pytorch_lightning.callbacks import ModelCheckpoint  # older versions

from neurosymbolic_model import SafeWithScallopLitModel
from ppe_dm_strict_prefilter import PPEDataModule

seed_everything(42)

# --- Data ---
dm = PPEDataModule(
    root_dir="../data/ppe-clean",
    img_size=224,
    batch_size=8,
    num_workers=8,
    train_augment=True,
    person_margin=0,
    min_person_area_frac=0.0,
)

# --- Model ---
model = SafeWithScallopLitModel(
    lr=1e-3,
    finetune_resnet=True,
    aux_weight=0,
    provenance="diffminmaxprob", #"diffaddmultprob", "addmultprob",
)

# --- Checkpointing: save after EVERY epoch (plus a 'last.ckpt') ---
ckpt_cb = ModelCheckpoint(
    dirpath="../checkpoints_v8_neuro_no_aux_loss/safe",
    filename="safe-{epoch:02d}-train={train_loss:.4f}-val={val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=-1,
    save_on_train_epoch_end=False,
    save_last=True
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
