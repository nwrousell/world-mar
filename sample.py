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
import cv2
import numpy as np

from world_mar.modules.utils import instantiate_from_config
from world_mar.dataset.dataloader import MinecraftDataModule, MinecraftDataset
from world_mar.models.mar import WorldMAR

from train import find_latest_checkpoint

from world_mar.oasis_utils.vae import format_image

def main(args):
    cfg = OmegaConf.load(args.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    ckpt_path = find_latest_checkpoint(args.resume)
    model_params = cfg.model.params
    model = WorldMAR.load_filtered_checkpoint(ckpt_path, **model_params, map_location="cpu")
    model.eval()
    model.to(device)

    # frame_path = "data/cheeky-cornflower-setter-db485e7cdf63-20220416-103055/frame000002.jpg"
    # frame = torch.tensor(cv2.imread(frame_path).astype(np.float32) / 255)
    # frame = frame[..., [2, 1, 0]] # BGR --> RGB

    # latent = model.vae.encode(frame.unsqueeze(0)).sample()

    # sampled_frames = model.vae.decode(latent)

    # save_image(sampled_frames[0], "ref.png")

    # get data
    dataloader = MinecraftDataModule(dataset_dir=args.data_dir, batch_sz=3).val_dataloader()

    batch = next(iter(dataloader))
    latents = batch["frames"].to(device)
    batch_nframes = batch["num_non_padding_frames"].to(device)
    actions = batch["action"].to(device)
    poses = batch["plucker"].to(device)
    timestamps = batch["timestamps"].to(device)

    print(latents.dtype, latents.max(), latents.min())

    # sample latent
    sampled_latents = model.sample(latents, actions, poses, timestamps, batch_nframes)

    # decode to frame
    sampled_frames = ((model.vae.decode(sampled_latents).clip(-1,1) + 1) / 2) * 255 # 1, 3, 360, 640
    print("sampled", sampled_frames.shape)

    # write to file
    save_image(sampled_frames[0], "sample.png")
    

if __name__ == "__main__":
    parser = ArgumentParser()
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
    
    args = parser.parse_args()
    main(args)