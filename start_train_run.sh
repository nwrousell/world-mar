#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -n 4                    # number of cores
#SBATCH -p gpu --gres=gpu:a5000:2
#SBATCH --mem=60G               # total memory per node (4 GB per cpu-core is default)
#SBATCH -t 24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=noah_rousell@brown.edu

set -e

module purge
unset LD_LIBRARY_PATH

export PYTHONUNBUFFERED=TRUE

source env/bin/activate

export HOME=/users/nrousell 

unset SLURM_NTASKS
python train.py -n $1 -c configs/world_mar.yaml -d /users/nrousell/scratch/minecraft-raw