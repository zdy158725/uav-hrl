#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=3 /home/dfc/.conda/envs/hhmarl_diffusion/bin/python -c '
from exp_diffusion_hrl.raytune import train_diffusion_hier_from_baseline

print(train_diffusion_hier_from_baseline(
    seed=0,
    num_cpus=8,
    num_gpus=1,
    num_env_runners=2,
    learner_gpus=0.2,
    worker_gpus=0.4,
    training_iterations=80,
    checkpoint_frequency=20,
))
'
