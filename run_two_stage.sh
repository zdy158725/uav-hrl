#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dfc/.conda/envs/hhmarl/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
SEED="${SEED:-0}"
NUM_CPUS="${NUM_CPUS:-8}"
NUM_GPUS="${NUM_GPUS:-0}"
NUM_ENV_RUNNERS="${NUM_ENV_RUNNERS:-2}"
POLICY_DIR="${POLICY_DIR:-$PROJECT_DIR/outputs/policies}"

cd "$PROJECT_DIR"

echo "项目目录: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "seed=$SEED num_cpus=$NUM_CPUS num_gpus=$NUM_GPUS num_env_runners=$NUM_ENV_RUNNERS"
echo "policy_dir=$POLICY_DIR"

echo
echo "[1/5] 训练低层 attack 策略"
ATTACK_CKPT="$(
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from exp_article_test.raytune import train_low_level_attack; print(train_low_level_attack(seed=$SEED, num_cpus=$NUM_CPUS, num_gpus=$NUM_GPUS, num_env_runners=$NUM_ENV_RUNNERS))" | tail -n 1
)"
echo "attack checkpoint: $ATTACK_CKPT"

echo
echo "[2/5] 导出低层 attack 策略"
ATTACK_POLICY="$(
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from exp_article_test.raytune import export_low_level_policies; print(export_low_level_policies('$ATTACK_CKPT', mode='attack', output_dir='$POLICY_DIR', num_gpus=0))" | tail -n 1
)"
echo "attack policy: $ATTACK_POLICY"

echo
echo "[3/5] 训练低层 evade 策略"
EVADE_CKPT="$(
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from exp_article_test.raytune import train_low_level_evade; print(train_low_level_evade(seed=$SEED, num_cpus=$NUM_CPUS, num_gpus=$NUM_GPUS, num_env_runners=$NUM_ENV_RUNNERS))" | tail -n 1
)"
echo "evade checkpoint: $EVADE_CKPT"

echo
echo "[4/5] 导出低层 evade 策略"
EVADE_POLICY="$(
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from exp_article_test.raytune import export_low_level_policies; print(export_low_level_policies('$EVADE_CKPT', mode='evade', output_dir='$POLICY_DIR', num_gpus=0))" | tail -n 1
)"
echo "evade policy: $EVADE_POLICY"

echo
echo "[5/5] 使用冻结低层策略训练高层"
HIGH_LEVEL_CKPT="$(
  "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); from exp_article_test.raytune import train_hier_v1; print(train_hier_v1(seed=$SEED, num_cpus=$NUM_CPUS, num_gpus=$NUM_GPUS, num_env_runners=$NUM_ENV_RUNNERS, use_script_low_level=False, low_level_policy_dir='$POLICY_DIR'))" | tail -n 1
)"
echo "high-level checkpoint: $HIGH_LEVEL_CKPT"

echo
echo "两阶段训练完成。"
echo "attack policy: $ATTACK_POLICY"
echo "evade policy: $EVADE_POLICY"
echo "high-level checkpoint: $HIGH_LEVEL_CKPT"
