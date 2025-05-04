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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from world_mar.modules.utils import instantiate_from_config
from world_mar.dataset.dataloader import MinecraftDataModule

LOG_PARENT = "logs"

class ImageLogger(pl.Callback):
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0:
            num_to_sample = 4
            latents = batch["frames"].to(pl_module.device)[:num_to_sample]
            batch_nframes = batch["num_non_padding_frames"].to(pl_module.device)[:num_to_sample]
            actions = batch["action"].to(pl_module.device)[:num_to_sample]
            poses = batch["plucker"].to(pl_module.device)[:num_to_sample]
            timestamps = batch["timestamps"].to(pl_module.device)[:num_to_sample]
            
            # sample latents
            sampled_latents = pl_module.sample(latents, actions, poses, timestamps, batch_nframes) # n 576 16

            # decode to frames
            # to_decode = torch.cat([latents[:, 1], latents[:, 0], sampled_latents], dim=0)
            to_decode = torch.cat([latents[:, 3], latents[:, 2], latents[:, 1], latents[:, 0], sampled_latents], dim=0)
            pl_module.vae.to("cuda")
            with torch.autocast(device_type="cuda", enabled=False):
                one, two, three, four, five = (((pl_module.vae.decode(to_decode) + 1) / 2) * 255).to(torch.uint8).chunk(chunks=5, dim=0) # each are n c h w
            pl_module.vae.to("cpu")

            trifolds = torch.cat([one, two, three, four, five], dim=-1) # concat along width dim

            wandb_images = [wandb.Image(img) for img in trifolds]

            # Log to W&B
            trainer.logger.experiment.log({"generated_images": wandb_images}, step=global_step)

def get_callbacks(logdir):
    return [
        ModelCheckpoint(
            dirpath=os.path.join(logdir, "checkpoints/"),
            filename="epoch{epoch:02d}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        ImageLogger()
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

