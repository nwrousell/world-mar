import os
import datetime
import shutil

from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from world_mar.modules.utils import instantiate_from_config

LOG_PARENT = "logs"

def get_callbacks(logdir):
    return [
        ModelCheckpoint(
            dirpath=os.path.join(logdir, "checkpoints/"),
            filename="epoch{epoch:02d}",
            save_top_k=3,
            monitor="train_loss", # TODO: probably do val loss, im lazy
            mode="min"
        )
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
    train_data = ... # TODO: @noah make this a lightning dataloader    

    # load model
    model_cfg = cfg.model
    model = instantiate_from_config(model_cfg)

    # TODO: train!
    model.learning_rate = model_cfg.learning_rate
    # TODO: add some more callbacks i.e. sequence logging (image, etc.) feedback to this
    callbacks = get_callbacks(logdir)

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        accelerator = "gpu"
        model.learning_rate *= num_devices
    else:
        num_devices = 1
        accelerator = "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator, devices=num_devices, precision="bf16-mixed",
        callbacks=callbacks
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
   # ... add more here
    
    args = parser.parse_args()

    main(args)

