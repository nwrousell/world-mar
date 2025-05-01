#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -n 8                    # number of cores
#SBATCH -p gpu --gres=gpu:2
#SBATCH --mem=32G               # total memory per node (4 GB per cpu-core is default)
#SBATCH -t 48:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=noah_rousell@brown.edu

set -e

module purge
unset LD_LIBRARY_PATH

export PYTHONUNBUFFERED=TRUE

source env/bin/activate

# python -m world_mar.dataset.download --json-file world_mar/dataset/indexes/all_10xx_Jun_29.json --num-demos 1000 --output-dir ~/scratch/minecraft-raw
python -m world_mar.dataset.download --output-dir ~/scratch/minecraft-raw