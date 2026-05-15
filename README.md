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
