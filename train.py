import os
import datetime
import shutil
import torch

from argparse import ArgumentParser
from omegaconf import OmegaConf

LOG_PARENT = "logs"

def main(args):
    # loading conf
    cfg = OmegaConf.load(args.config)

    # set up logdir
    cfg_fname = os.path.splitext(os.path.split(args.config)[-1])[0]
    if args.name:
        name = args.name
    else:
        now = datetime.datetime.now().strftime("%m-%d-T%H-%M-%S")
        name = now + "_" + cfg_fname
        
    logdir = os.path.join(LOG_PARENT, name)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, os.path.join(logdir, os.path.basename(args.config)))

    # more shenanigans...

    # TODO: train!

    dataloader = ...
    vae = ...
    model = ...
    optimizer = ...

    for batch in dataloader:
        optimizer.zero_grad()
        
        # dataloader has already sampled context frames with high overlap and made poses relative
        frame_to_pred, context_frames, context_relative_poses, actions = batch

        all_frames = [frame_to_pred, context_frames] # stack these correctly
        latents = vae.encode(all_frames)
        context_latents, latent_to_pred = latents[0], latents[1:] # fix this

        pred_latent = model(context_latents, context_relative_poses, actions) # does ROPE and action embedding

        loss = diff_loss(pred_latent, latent_to_pred)
        loss.backward()

        optimizer.step()


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
    # ... add more here
    
    args = parser.parse_args()

    main(args)

