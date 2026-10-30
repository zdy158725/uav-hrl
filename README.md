# Two-Stage HRL Standalone

这是从 `air_combat_scene/10` 当前可运行版本中整理出来的独立项目。

## 特点

- 不依赖 `air_combat_scene/10` 目录本身
- 不依赖 `hhmarl_2D-main` 代码内容
- 输出默认写到当前项目目录下的 `outputs/`
- 支持：
  - 高层 HRL 训练
  - 低层 `attack` / `evade` 单独训练
  - 低层模型导出
  - 两阶段一键训练
  - 训练过程自动生成 `progress.csv` 和 `training_metrics.png`

## 目录

- `exp_article_test/`: 核心环境、场景、训练入口
- `main.py`: 高层训练入口示例
- `run_two_stage.sh`: 两阶段一键训练脚本
- `plot_progress_metrics.py`: 从 `progress.csv` 画训练曲线

## 低层动作库

低层动作库使用原来的 15 组 `[nx, nn, phi]` 控制量，动作编号为 `0～14`。脚本生成的连续控制量会匹配到其中最近的一组。动作总数由 `exp_article_test/maneuver.py` 的 `ACTION_COUNT` 定义。

原有的 15 动作低层模型与动作编号兼容；60 动作低层模型的输出空间与当前环境不兼容，需要重新训练或使用原有的 15 动作模型。高层策略的动作空间未变，但低层行为和脚本机动会改变其面对的环境，因此使用 60 动作低层模型训练的高层策略应重新训练或微调并重新评估。

## 交战结算

新训练的高层策略默认每 2 秒重新决策一次：战场物理步长为 0.1 秒，每次战场更新包含 10 个物理步，高层动作持续 2 次战场更新。可通过 `HRLConfig["sub_steps"]` 和 `DiffusionHRLConfig["sub_steps"]` 调整。用旧的 10 秒间隔训练的 checkpoint 评估时，`evaluate_fixed_checkpoints.py` 和 `evaluate_diffusion_hrl.py` 默认仍采用 10 秒；评估新模型时传入 `--sub-steps 2 --max-steps 251`。训练结束自动触发的固定评估会采用本次训练的间隔及 251 步上限，对应战场约 500 秒的最大仿真时间。调整决策间隔后应重新训练高层策略，再比较结果。

每个 0.1 秒物理步仅让本步选择 `attack` 的飞机计算伤害；选择 `evade` 的飞机仍可受伤，但不会造成伤害。蓝方脚本只有选择追击时才具备攻击资格。战场先收集红蓝双方的命中与伤害，再统一扣血和判定击毁，因此双方可在同一物理步互相击毁。低层单独训练时，红方攻击资格由训练模式 `attack` 或 `evade` 决定。

攻击伤害只针对已选定的敌机计算。红方高层在每次决策开始时按飞机 ID 锁定目标，低层机动与伤害结算使用同一目标；候选敌机距离排序变化不会切换目标。目标被击毁或未满足命中条件时，本次不向其他敌机转移伤害，下一次高层决策再选择。蓝方脚本使用其追击目标，低层单独训练使用已分配的训练目标。

在现有两模式交战中，未选择进攻的飞机受到攻击时默认只承受 50% 伤害；进攻中的飞机承受全部伤害。比例由 `exp_article_test/config.py` 的 `BattleFieldConfig["evade_damage_multiplier"]` 设置，范围为 0～1。减伤只作用于攻击伤害，不改变碰撞和出界规则。

物理步长仍为 0.1 秒，但同一架飞机两次成功造成伤害之间默认至少间隔 0.5 秒。未命中不会开始冷却。可通过 `BattleFieldConfig["fire_cooldown_seconds"]` 调整；设为 `0` 可关闭间隔限制。

## 快速开始

先进入项目目录：

```bash
cd /home/dfc/Documents/zdy_documents/air_combat_scene/uav_hrl
```

高层 baseline：

```bash
/home/dfc/.conda/envs/hhmarl/bin/python main.py
```

两阶段训练：

```bash
./run_two_stage.sh
```

若要在当前环境下重新训练 15 动作低层和 2 秒决策高层，并固定蓝方为旧 π0（`checkpoint_000140` 高层策略及其原有低层模型），使用：

```bash
CUDA_VISIBLE_DEVICES=2 /home/dfc/.conda/envs/hhmarl/bin/python train_two_stage_vs_pi0.py --num-gpus 1 --num-cpus 8 --num-env-runners 2
```

脚本依次训练 `attack`、`evade`，导出到独立的新策略目录，再训练高层；蓝方保持 10 秒一次高层决策，红方保持当前的 2 秒。默认每阶段 1000 次 PPO 迭代、每 20 次保存检查点。可先加 `--dry-run` 验证模型加载和环境交互，不会启动训练。

自定义参数：

```bash
NUM_CPUS=4 NUM_ENV_RUNNERS=1 SEED=1 ./run_two_stage.sh
```

## 输出

默认输出都在当前项目的 `outputs/` 下：

- `outputs/checkpoints/`
- `outputs/low_level/`
- `outputs/policies/`
- `outputs/trajectories/`

每次训练目录内会自动生成：

- `progress.csv`
- `training_metrics.png`
- `checkpoint_*`
