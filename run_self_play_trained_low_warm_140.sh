#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PYTHON_BIN="${PYTHON_BIN:-/home/dfc/.conda/envs/hhmarl/bin/python}"
POLICY_DIR="${POLICY_DIR:-$(pwd)/outputs/policies}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$(pwd)/outputs/checkpoints/PPO_SELF_PLAY_WARM_2026-06-25_17-52-01/checkpoint_000140}"
TRAINING_ITERATIONS="${TRAINING_ITERATIONS:-80}"

if [[ ! -f "$POLICY_DIR/low_attack_policy.pt" ]]; then
  echo "缺少 Attack 低层模型: $POLICY_DIR/low_attack_policy.pt" >&2
  exit 1
fi
if [[ ! -f "$POLICY_DIR/low_evade_policy.pt" ]]; then
  echo "缺少 Evade 低层模型: $POLICY_DIR/low_evade_policy.pt" >&2
  exit 1
fi
if [[ ! -d "$INIT_CHECKPOINT" ]]; then
  echo "缺少高层初始 checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
fi

echo "高层初始策略: $INIT_CHECKPOINT"
echo "冻结低层策略目录: $POLICY_DIR"
echo "Attack: checkpoint_000060 -> low_attack_policy.pt"
echo "Evade:  checkpoint_000100 -> low_evade_policy.pt"
echo "训练规模: 3v3"
echo "自博弈新训练轮数: $TRAINING_ITERATIONS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

"$PYTHON_BIN" -u - <<PY
from exp_article_test.config import RayCheckpointConfig
from exp_article_test.raytune import train_hier_self_play_from_checkpoint

RayCheckpointConfig["training_iteration"] = int("$TRAINING_ITERATIONS")

print(
    train_hier_self_play_from_checkpoint(
        init_checkpoint="$INIT_CHECKPOINT",
        seed=0,
        num_cpus=8,
        num_gpus=1,
        num_env_runners=2,
        use_script_low_level=False,
        low_level_policy_dir="$POLICY_DIR",
        reset_pool=True,
    )
)
PY
