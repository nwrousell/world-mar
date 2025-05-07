import os
import datetime
import shutil
import torch

from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from torchvision.io import read_image
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from world_mar.modules.utils import instantiate_from_config
from world_mar.dataset.dataloader import MinecraftDataModule

LOG_PARENT = "logs"

class ImageLogger(pl.Callback):
    def __init__(self, log_every_n_steps=500):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps != 0:
            return
        
        num_to_sample = 4
        latents = batch["frames"].to(pl_module.device)[:num_to_sample]
        batch_nframes = batch["num_non_padding_frames"].to(pl_module.device)[:num_to_sample]
        actions = batch["action"].to(pl_module.device)[:num_to_sample]
        poses = batch["plucker"].to(pl_module.device)[:num_to_sample]
        timestamps = batch["timestamps"].to(pl_module.device)[:num_to_sample]
        
        # sample latents
        sampled_latents = pl_module.sample(latents, actions, poses, timestamps, batch_nframes) # n 576 16

        # decode to frames
        to_decode = torch.cat([latents[:, 3], latents[:, 2], latents[:, 1], latents[:, 0], sampled_latents], dim=0)
        pl_module.vae.to("cuda")
        with torch.autocast(device_type="cuda", enabled=False):
            one, two, three, four, five = (
                ((pl_module.vae.decode(to_decode).clip(-1, 1) + 1) / 2) * 255
            ).to(torch.uint8).chunk(chunks=5, dim=0)
        pl_module.vae.to("cpu")

        trifolds = torch.cat([one, two, three, four, five], dim=-1) # concat along width dim

        wandb_images = [wandb.Image(img) for img in trifolds]

        # Log to W&B
        trainer.logger.experiment.log({"generated_images": wandb_images}, step=global_step)

class MaskedImageLogger(pl.Callback):
    def __init__(self, log_every_n_steps=500):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps != 0:
            return

        num_to_sample = 4
        device        = pl_module.device
        latents       = batch["frames"].to(device)[:num_to_sample]
        batch_nframes = batch["num_non_padding_frames"].to(device)[:num_to_sample]
        actions       = batch["action"].to(device)[:num_to_sample]
        poses         = batch["plucker"].to(device)[:num_to_sample]
        timestamps    = batch["timestamps"].to(device)[:num_to_sample]

        # sample latents
        sampled_latents = pl_module.sample(latents, actions, poses, timestamps, batch_nframes)

        # decode to frames
        to_decode = torch.cat([latents[:, 3], latents[:, 2], latents[:, 1], latents[:, 0], sampled_latents], dim=0)
        pl_module.vae.to("cuda")
        with torch.autocast(device_type="cuda", enabled=False):
            one, two, three, four, five = (
                ((pl_module.vae.decode(to_decode).clip(-1, 1) + 1) / 2) * 255
            ).to(torch.uint8).chunk(chunks=5, dim=0)
        pl_module.vae.to("cpu")

        # retrieve token masks and prediction index from forward pass
        pred_mask = pl_module._last_pred_mask[:num_to_sample]  # (B, H_patch*W_patch)
        ctx_mask = pl_module._last_ctx_mask[:num_to_sample]    # (B, P-1, H_patch*W_patch)
        pred_idx = pl_module._last_pred_idx                    # int

        # construct mask going through entire previous nom-memory context frame window
        ctx_mask_left = ctx_mask[:, :pred_idx, :]
        ctx_mask_middle = torch.zeros_like(pred_mask, dtype=torch.bool)
        ctx_mask_right = ctx_mask[:, pred_idx:, :]
        ctx_mask = torch.cat([ctx_mask_left, ctx_mask_middle, ctx_mask_right], dim=-2)  # (B, P, H_patch*W_patch)

        # compute pixel‐patch dims
        B, C, H_pix, W_pix = five.shape
        latent_h, latent_w = pl_module.vae_seq_h, pl_module.vae_seq_w
        pix_per_lat_h = H_pix // latent_h
        pix_per_lat_w = W_pix // latent_w
        p = pl_module.patch_size
        H_patch = latent_h // p
        W_patch = latent_w // p

        # white-out masked tokens on pred frame
        decoded = [one, two, three, four, five]
        pred_mask = pred_mask.reshape(B, H_patch, W_patch)
        for i in range(B):
            rows, cols = pred_mask[i].nonzero(as_tuple=True)
            for r, c in zip(rows.tolist(), cols.tolist()):
                # lower-left and upper-right patch corners
                y0 =   r   * p * pix_per_lat_h
                y1 = (r+1) * p * pix_per_lat_h
                x0 =   c   * p * pix_per_lat_w
                x1 = (c+1) * p * pix_per_lat_w
                # whiteout the whole patch
                five[i,:, y0:y1, x0:x1] = 255

        # black-out masked tokens on other previous non-memory context frames
        P = ctx_mask.shape[1]
        ctx_mask = ctx_mask.reshape(B, P, H_patch, W_patch)
        for p_idx in range(P):
            mask_p = ctx_mask[:, p_idx, :, :]  # (B, H_patch, W_patch)
            decoded_idx = (len(decoded) - 1) - p_idx
            for i in range(B):
                rows, cols = mask_p[i].nonzero(as_tuple=True)
                for r, c in zip(rows.tolist(), cols.tolist()):
                    # lower-left and upper-right patch coners
                    y0 =   r   * p * pix_per_lat_h
                    y1 = (r+1) * p * pix_per_lat_h
                    x0 =   c   * p * pix_per_lat_w
                    x1 = (c+1) * p * pix_per_lat_w
                    # blackout the whole patch
                    decoded[decoded_idx][i, :, y0:y1, x0:x1] = 0

        # log to wandb
        trifolds_masked = torch.cat([one, two, three, four, five], dim=-1)
        wandb_imgs = [wandb.Image(img) for img in trifolds_masked]
        trainer.logger.experiment.log({"generated_images_masked": wandb_imgs}, step=global_step)

def get_callbacks(logdir):
    return [
        ModelCheckpoint(
            dirpath=os.path.join(logdir, "checkpoints/"),
            filename="epoch{epoch:02d}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        ImageLogger(),
        MaskedImageLogger(),
        LearningRateMonitor(logging_interval='step')
    ]

def find_latest_checkpoint(logdir):
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"no checkpoint directory found in {checkpoint_dir}")
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    latest_ckpt = max(ckpts, key=os.path.getctime)
    return latest_ckpt

def main(args):
    # loading conf
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config: {args.config}")

    # set up logdir
    if args.name:
        name = args.name
    else:
        cfg_fname = os.path.splitext(os.path.split(args.config)[-1])[0]
        now = datetime.datetime.now().strftime("%m-%d-T%H-%M-%S")
        name = now + "_" + cfg_fname

    if not args.resume:
        logdir = os.path.join(LOG_PARENT, name)
        os.makedirs(logdir, exist_ok=True)
        shutil.copy(args.config, os.path.join(logdir, os.path.basename(args.config)))
        ckpt_path = None
    else:
        logdir = args.resume
        ckpt_path = find_latest_checkpoint(args.resume)
    print(f"Initialized logging directory: {logdir}")

    # load model and dataloader from config
    model_cfg = cfg.model
    # model_cfg.params.vae_config = None # destroy vae_config so it doesn't load the vae
    model = instantiate_from_config(model_cfg)
    model.vae.to("cpu")
    print("Loaded model")
    dataloader_cfg = cfg.dataloader
    if args.data_dir:
        dataloader_cfg.dataset_dir = args.data_dir
    dataloader = instantiate_from_config(dataloader_cfg)
    print("Loaded dataloader")

    # set the learning rate of the model
    model.learning_rate = model_cfg.learning_rate
    print(f"Set initial learning rate to: {model.learning_rate}")

    # get the callback functions for each batch
    callbacks = get_callbacks(logdir)

    if torch.cuda.is_available():
        print("CUDA available; printing GPU info:")
        num_devices = torch.cuda.device_count()
        accelerator = "gpu"
        model.learning_rate *= num_devices
        #torch.set_float32_matmul_precision('medium')
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA unavailable; running on CPU.")
        num_devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        #strategy="ddp",
        accelerator=accelerator, devices=num_devices, precision="bf16-mixed",
        check_val_every_n_epoch=1,
        callbacks=callbacks, logger=WandbLogger(project="WorldMar", log_model="all", name=name, entity="praccho-brown-university")
    )

    print("Starting training...")
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path = ckpt_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="experiment name"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config"
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="",
        type=str,
        help="path to data dir"
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        help="path to previous run's logdir"
    )
    
    args = parser.parse_args()

    main(args)