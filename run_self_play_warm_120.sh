#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

/home/dfc/.conda/envs/hhmarl/bin/python -u - <<'PY'
import sys

sys.path.insert(0, ".")

from exp_article_test.raytune import (
    RayCheckpointConfig,
    train_hier_self_play_from_checkpoint,
)

RayCheckpointConfig["training_iteration"] = 80

print(
    train_hier_self_play_from_checkpoint(
        init_checkpoint="/home/dfc/Documents/zdy_documents/air_combat_scene/uav_hrl/outputs/checkpoints/PPO_2026-06-25_15-08-10_resume_2026-06-25_16-48-48/checkpoint_000120",
        seed=0,
        num_cpus=8,
        num_gpus=1,
        num_env_runners=2,
        reset_pool=True,
    )
)
PY
