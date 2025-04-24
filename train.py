import os
import datetime
import shutil

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

