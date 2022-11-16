#!/bin/bash

#SBATCH
#SBATCH --mail-type=END                      # Request status by email
#SBATCH --mail-user=yzs2@cornell.edu         # Email address to send results to.
#SBATCH --job-name=glow_mnist                # Job name
#SBATCH --output=%x_%j.out                   # Output file: %x is job name, %j is job id
#SBATCH --error=%x_%j.err                    # Error file: %x is job name, %j is job id
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                # Total number of cores requested
#SBATCH --get-user-env                       # Retrieve the users login environment
#SBATCH --mem=75G                            # Server memory requested (per node; 1000M ~= 1G)
#SBATCH --gres=gpu:8                         # Type/number of GPUs needed
#SBATCH --partition=kuleshov,gpu             # Request partition (gpu==medium priority; kuleshov==high priority)
#SBATCH --time=24:00:00                      # Set max runtime for job
#SBATCH --requeue                            # Requeue job


export PYTHONPATH="${PWD}"

# shellcheck source=${HOME}/.bashrc
source "${CONDA_SHELL}"
conda activate glow_v2

mpiexec -n 8 python train.py \
  --problem mnist \
  --image_size 32 \
  --n_level 2 \
  --depth 2 \
  --flow_permutation 2 \
  --flow_coupling 1 \
  --seed 0 \
  --lr 0.001 \
  --gradient_checkpointing 0 \
  --epochs_full_valid 1 \
  --epochs 500 \
  --logdir ./mnist_logs \
  --restore_path ./mnist_logs/model_latest.ckpt
conda deactivate
