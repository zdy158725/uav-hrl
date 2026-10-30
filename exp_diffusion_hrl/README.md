# Diffusion-HRL

该包在不修改 `exp_article_test/` 的前提下，为红方高层策略增加敌机未来轨迹预测。

## 数据流

1. 从底层0.1秒记录中提取最近30步六机结构化状态。
2. EMA Diffusion模型以10步DDIM批量预测全部存活蓝机未来30步相对轨迹。
3. 两个候选目标分别提取第10/20/30步相对位置和 `ready` 标记。
4. 新特征与原38维观测拼接为58维，高层仍输出三种宏动作。
5. Attack-60与Evade-100低层策略保持冻结，蓝方使用增强脚本。

## 运行

预测器artifact已导出到 `outputs/diffusion_hrl/predictor/predictor.pt`。正式训练：

```bash
bash scripts/run_diffusion_hrl_gpu3.sh
```

训练结果位于 `outputs/diffusion_hrl/checkpoints/`，每20轮保存checkpoint并更新训练曲线。

评估示例：

```bash
CUDA_VISIBLE_DEVICES=3 /home/dfc/.conda/envs/hhmarl_diffusion/bin/python \
  evaluate_diffusion_hrl.py \
  --checkpoint outputs/diffusion_hrl/checkpoints/<run>/checkpoint_000080 \
  --episodes 500 \
  --output-dir outputs/diffusion_hrl/evaluation/<run>
```

评估覆盖增强脚本与旧 `pi0`，并真正交换策略控制阵营：分别让Diffusion策略控制红方和蓝方。输出同时包含含/不含平局胜率、存活数、奖励、攻防比例、ADE/FDE、ready比例和推理延迟。

离线预测验证默认采集baseline的100局轨迹，并与匀速外推比较：

```bash
CUDA_VISIBLE_DEVICES=3 /home/dfc/.conda/envs/hhmarl_diffusion/bin/python \
  evaluate_diffusion_predictor.py --episodes 100
```
