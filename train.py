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
    def __init__(self, log_every_n_steps=1000):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step % self.log_every_n_steps == 0:
            
            images = pl_module.sample_images() 

            # Convert to W&B Image format
            wandb_images = [wandb.Image(img) for img in images]

            # Log to W&B
            trainer.logger.experiment.log({"generated_images": wandb_images}, step=global_step)

def get_callbacks(logdir):
    return [
        ModelCheckpoint(
            dirpath=os.path.join(logdir, "checkpoints/"),
            filename="epoch{epoch:02d}",
            save_top_k=3,
            monitor="train_loss", # TODO: probably do val loss, im lazy
            mode="min"
        ),
        # ImageLogger()
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

    # more shenanigans...
    train_data = MinecraftDataModule(dataset_dir=args.data_dir)

    # load model
    model_cfg = cfg.model
    model_cfg.params.vae_config = None # destroy vae_config so it doesn't load the vae
    model = instantiate_from_config(model_cfg)

    model.learning_rate = model_cfg.learning_rate
    callbacks = get_callbacks(logdir)

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        accelerator = "gpu"
        model.learning_rate *= num_devices
        #torch.set_float32_matmul_precision('medium')
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        #strategy="ddp",
        accelerator=accelerator, devices=num_devices, precision="bf16-mixed",
        callbacks=callbacks, logger=WandbLogger(project="WorldMar", log_model="all", name=name, entity="praccho-brown-university")
    )

    trainer.fit(model, train_dataloaders=train_data, ckpt_path = ckpt_path)

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
        "-r",
        "--resume",
        default="",
        type=str,
        help="path to previous run's logdir"
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="",
        type=str,
        help="path to data directory"
    )
   # ... add more here
    
    args = parser.parse_args()

    main(args)

