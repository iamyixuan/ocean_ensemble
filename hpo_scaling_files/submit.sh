#!/bin/bash
#SBATCH -A m4259 
#SBATCH --job-name=dh_cbo
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
# #SBATCH --gpus-per-task=2


# User Configuration
INIT_SCRIPT=$PWD/load_modules.sh

SLURM_JOBSIZE=32
RANKS_PER_NODE=4

# Initialization of environment
source $INIT_SCRIPT

# change to the directory where this script is located
cd ../

srun -N $SLURM_JOBSIZE -n $(( $SLURM_JOBSIZE * $RANKS_PER_NODE )) python hpo_nll.py


echo "Complete"
