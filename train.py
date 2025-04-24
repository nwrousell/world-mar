import os
import datetime

from argparse import ArgumentParser
from omegaconf import OmegaConf

def main(args):
    # loading conf
    cfg = OmegaConf.load(args.config)

    # set up logdir
    cfg_fname = os.path.splitext(os.path.split(args.config)[-1])[0]
    now = datetime.datetime.now().strftime("%dT%H-%M-%S")
    nowname = now + "_" + cfg_fname
    logdir = os.path.join("logs", nowname)
    os.makedirs(logdir, exist_ok=True)

    # more shenanigans...

    # TODO: train!

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        required=True,
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

