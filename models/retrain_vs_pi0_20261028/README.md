# 15 动作分层策略：对战 π0 重新训练

这组参数对应 `train_two_stage_vs_pi0.py` 在 2026-10-28 启动的三阶段训练。随机种子为 0，进攻低层、逃逸低层和高层各训练 1000 轮。红方高层每约 2 秒决策一次；固定蓝方 π0 每约 10 秒决策一次。

| 文件 | 用途 | 原始训练产物 |
| --- | --- | --- |
| `low_attack_policy.pt` | 红方进攻低层策略 | `outputs/low_level/attack/PPO_2026-10-28_13-47-16/checkpoint_001000` 导出 |
| `low_evade_policy.pt` | 红方逃逸低层策略 | `outputs/low_level/evade/PPO_2026-10-28_16-35-57/checkpoint_001000` 导出 |
| `high_level_checkpoint/` | 红方高层 PPO 策略及 RLlib checkpoint 元数据 | `outputs/checkpoints/PPO_2026-10-28_19-32-52/checkpoint_001000` |
| `pi0/high_level_checkpoint/` | 训练时固定的蓝方 π0 高层策略 | `outputs/checkpoints/PPO_SELF_PLAY_WARM_2026-06-25_17-52-01/checkpoint_000140` |
| `pi0/policies/` | 训练时固定的蓝方 π0 低层策略 | `outputs/baselines/hrl_3v3_selfplay_ckpt220_lowppo_v1/policies` |

在仓库根目录运行下面的命令可以用这一组参数评估红方对战 π0；评估结果写入已被 Git 忽略的 `outputs/`：

```bash
python evaluate_fixed_checkpoints.py \
  --run-dir models/retrain_vs_pi0_20261028 \
  --checkpoints models/retrain_vs_pi0_20261028/high_level_checkpoint \
  --baseline-checkpoint models/retrain_vs_pi0_20261028/pi0/high_level_checkpoint \
  --opponents initial_policy \
  --use-trained-low-level \
  --low-level-policy-dir models/retrain_vs_pi0_20261028 \
  --blue-low-level-policy-dir models/retrain_vs_pi0_20261028/pi0/policies \
  --sub-steps 2 \
  --blue-high-level-sub-steps 10 \
  --max-steps 251 \
  --episodes 200 \
  --output-dir outputs/evaluation/retrain_vs_pi0_20261028
```

模型使用 PyTorch 和 Ray RLlib 的序列化格式，需要项目对应的 Python 依赖才能加载。`high_level_checkpoint/` 中的 `red_policy` 是该轮新训练的高层策略；`pi0/` 仅作为固定对手参数保留。
