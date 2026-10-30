#!/usr/bin/env bash
set -euo pipefail

UAV_HRL_ROOT="/mnt/disk2/zdy_documents/air_combat_scene/uav_hrl"
DIFFUSION_ROOT="/mnt/disk2/zdy_documents/diffusion_policy/uav_task_bundle"
SOURCE_DATASET="${UAV_HRL_ROOT}/outputs/diffusion_hrl/adaptation_dataset_7ckpt_ep50"
TARGET_DATASET="${UAV_HRL_ROOT}/outputs/diffusion_hrl/adaptation_dataset_7ckpt_ep100"
TRAIN_OUTPUT="${DIFFUSION_ROOT}/outputs/hrl_waypoint_retrain/cv_residual_7ckpt_ep100_lr3e-5_ep100"
LOG_DIR="${UAV_HRL_ROOT}/logs"

mkdir -p "${TARGET_DATASET}" "${LOG_DIR}"

if [ ! -f "${TARGET_DATASET}/aircombat_uav_3d_hrl_adaptation.csv" ]; then
  /home/dfc/.conda/envs/hhmarl_diffusion/bin/python - <<PY
from pathlib import Path
import shutil

source = Path("${SOURCE_DATASET}")
target = Path("${TARGET_DATASET}")
target.mkdir(parents=True, exist_ok=True)
for name in [
    "aircombat_uav_3d_hrl_adaptation.csv",
    "uav_3d_preprocess_stats.json",
]:
    src = source / name
    dst = target / name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"seeded {dst}")
PY
fi

cd "${UAV_HRL_ROOT}"
CUDA_VISIBLE_DEVICES=3 /home/dfc/.conda/envs/hhmarl_diffusion/bin/python \
  collect_diffusion_adaptation_dataset.py \
  --candidate-csv outputs/diffusion_hrl/adaptation_checkpoint_candidates.csv \
  --output-dir outputs/diffusion_hrl/adaptation_dataset_7ckpt_ep100 \
  --episodes-per-checkpoint 100 \
  --max-steps 120 \
  --seed 20260713 \
  --prediction-device cuda \
  --resume \
  2>&1 | tee "${LOG_DIR}/collect_adaptation_7ckpt_ep100.log"

cd "${DIFFUSION_ROOT}"
CUDA_VISIBLE_DEVICES=3 /home/dfc/.conda/envs/diffusion_policy/bin/python train.py \
  --config-name=uav_aircombat_lowdim_30hist_30future_structured_cv_residual_waypoint \
  hydra.run.dir="${TRAIN_OUTPUT}" \
  training.device=cuda:0 \
  training.num_epochs=100 \
  training.lr=3e-5 \
  training.val_every=5 \
  training.checkpoint_every=5 \
  dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  task.dataset.csv_path="${TARGET_DATASET}/aircombat_uav_3d_hrl_adaptation.csv" \
  task.dataset.split_stats_path="${TARGET_DATASET}/uav_3d_preprocess_stats.json" \
  2>&1 | tee "${LOG_DIR}/train_cv_residual_waypoint_7ckpt_ep100.log"
