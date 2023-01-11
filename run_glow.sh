#!/bin/bash

export PYTHONPATH="${PWD}"

# shellcheck source=${HOME}/.bashrc
source "${CONDA_SHELL}"
conda activate glow_v2

mpiexec -n 1 python train.py \
  --problem cifar10 \
  --image_size 32 \
  --n_level 3 \
  --depth 1 \
  --flow_permutation 2 \
  --flow_coupling 1 \
  --seed 0 \
  --lr 0.001 \
  --gradient_checkpointing 0 \
  --epochs_full_valid 1 \
  --logdir ./logs \
  --energy_distance
