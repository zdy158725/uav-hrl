import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from tqdm.auto import tqdm

from exp_article_test.config import RayModelConfig, RayTrainingConfig
from exp_diffusion_hrl.callback import DiffusionAlgoCallbacks
from exp_diffusion_hrl.config import DiffusionHRLConfig
from exp_diffusion_hrl.env import DiffusionAugmentedBattleEnv

try:
    from plot_progress_metrics import (
        plot_high_level_decisions,
        plot_metrics,
    )
except ModuleNotFoundError:
    plot_high_level_decisions = plot_metrics = None


PROGRESS_COLUMNS = [
    "training_iteration", "episode_reward_mean", "episode_reward_min",
    "episode_reward_max", "episode_len_mean", "episodes_this_iter", "episodes_total",
    "time_this_iter_s", "time_total_s", "custom_metrics/log_red_win_mean_mean",
    "custom_metrics/log_red_win_no_draw_mean_mean", "custom_metrics/log_red_alive_mean",
    "custom_metrics/log_blue_alive_mean", "custom_metrics/log_attack_ratio_mean",
    "custom_metrics/log_evade_ratio_mean", "custom_metrics/prediction_ready_ratio_mean",
    "custom_metrics/prediction_latency_ms_mean",
]


def env_creator(config):
    return DiffusionAugmentedBattleEnv(env_config=config)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    del episode, worker, kwargs
    if agent_id.startswith("red_"):
        return "red_policy"
    raise ValueError(f"Unexpected agent id: {agent_id}")


def _nested(result, key, default=""):
    current = result
    for part in key.split("/"):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _build_config(env_config, learner_gpus, worker_gpus, num_env_runners):
    return (
        PPOConfig()
        .environment(
            env="DiffusionAugmentedBattleEnv", disable_env_checking=True,
            env_config=env_config,
        )
        .framework("torch")
        .resources(
            num_gpus=float(learner_gpus), num_cpus_per_worker=1,
            num_gpus_per_worker=float(worker_gpus),
        )
        .rollouts(num_rollout_workers=int(num_env_runners))
        .training(
            lr=RayTrainingConfig["learning_rate"],
            clip_param=RayTrainingConfig["clip_param"],
            gamma=RayTrainingConfig["gamma"],
            train_batch_size=RayTrainingConfig["train_batch_size"],
            sgd_minibatch_size=RayTrainingConfig["sgd_minibatch_size"],
            num_sgd_iter=RayTrainingConfig["num_sgd_iter"],
            entropy_coeff=RayTrainingConfig["entropy_coeff"],
            vf_loss_coeff=RayTrainingConfig["vf_loss_coeff"],
            model=RayModelConfig["model"],
            use_gae=True,
            use_critic=True,
        )
        .multi_agent(
            policies={"red_policy": PolicySpec()},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["red_policy"],
        )
        .callbacks(callbacks_class=DiffusionAlgoCallbacks)
    )


def migrate_baseline_policy(algo, baseline_bundle):
    source_path = Path(baseline_bundle) / "high_level_policy"
    source_policy = Policy.from_checkpoint(str(source_path))
    source = source_policy.get_weights()
    target_policy = algo.get_policy("red_policy")
    target = target_policy.get_weights()
    migrated = {}
    expanded_keys = {
        "_hidden_layers.0._model.0.weight",
        "_value_branch_separate.0._model.0.weight",
    }
    for key, target_value in target.items():
        if key not in source:
            raise KeyError(f"Baseline policy is missing weight: {key}")
        source_value = source[key]
        if key in expanded_keys:
            if source_value.shape[1] != 38 or target_value.shape[1] != 58:
                raise ValueError(
                    f"Unexpected migration shape for {key}: {source_value.shape} -> {target_value.shape}"
                )
            value = np.zeros_like(target_value)
            value[:, :38] = source_value
            migrated[key] = value
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(f"Shape mismatch for {key}: {source_value.shape} != {target_value.shape}")
            migrated[key] = source_value.copy()
    target_policy.set_weights(migrated)

    check = target_policy.get_weights()
    for key in expanded_keys:
        np.testing.assert_allclose(check[key][:, :38], source[key], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(check[key][:, 38:], 0.0, atol=0.0, rtol=0.0)
    source_device = next(source_policy.model.parameters()).device
    target_device = next(target_policy.model.parameters()).device
    old_obs = torch.linspace(
        -1.0, 1.0, 38, dtype=torch.float32, device=source_device
    ).reshape(1, 38)
    target_old_obs = old_obs.to(target_device)
    new_obs = torch.cat(
        [target_old_obs, torch.zeros((1, 20), dtype=torch.float32, device=target_device)],
        dim=1,
    )
    source_policy.model.eval()
    target_policy.model.eval()
    with torch.no_grad():
        source_logits, _ = source_policy.model({"obs": old_obs}, [], None)
        source_value = source_policy.model.value_function().clone()
        target_logits, _ = target_policy.model({"obs": new_obs}, [], None)
        target_value = target_policy.model.value_function().clone()
    logits_error = float(
        torch.max(torch.abs(source_logits.cpu() - target_logits.cpu())).item()
    )
    value_error = float(
        torch.max(torch.abs(source_value.cpu() - target_value.cpu())).item()
    )
    if logits_error >= 1e-5 or value_error >= 1e-5:
        raise AssertionError(
            f"Migration parity failed: logits={logits_error}, value={value_error}"
        )
    return {
        "source": str(source_path), "expanded_keys": sorted(expanded_keys),
        "max_logits_error": logits_error, "max_value_error": value_error,
    }


def _write_progress(path, result, iteration):
    row = {
        column: result.get(column, iteration) if column == "training_iteration"
        else _nested(result, column, "")
        for column in PROGRESS_COLUMNS
    }
    with path.open("a", newline="") as file:
        csv.DictWriter(file, fieldnames=PROGRESS_COLUMNS).writerow(row)


def _update_plots(progress_path, run_dir):
    if plot_metrics is None:
        return
    try:
        plot_metrics(progress_path, run_dir / "training_metrics.png")
        if plot_high_level_decisions is not None:
            plot_high_level_decisions(progress_path, run_dir / "high_level_decisions.png")
    except Exception as exc:
        print(f"训练曲线更新失败，已跳过: {exc}")


def _rollout_for_trajectory(algo, env, seed, max_steps=251):
    random.seed(seed)
    np.random.seed(seed)
    observations, _ = env.reset(seed=seed)
    red_agents = list(env.possible_agents)
    decisions = {agent: [] for agent in red_agents}
    low_actions = {agent: [] for agent in red_agents}
    segments = {agent: [] for agent in red_agents}
    for step_index in range(max_steps):
        if not observations:
            break
        starts = {f"red_{u.id}": len(u.record_pos) for u in env.scene.bf.red_uav}
        actions = {}
        for agent, observation in observations.items():
            result = algo.compute_single_action(observation, policy_id="red_policy", explore=False)
            action = result[0] if isinstance(result, tuple) else result
            actions[agent] = int(action)
        for agent in red_agents:
            decisions[agent].append(-1 if agent not in actions else int(actions[agent] > 0))
        observations, _, terminated, truncated, _ = env.step(actions)
        for red in env.scene.bf.red_uav:
            agent = f"red_{red.id}"
            end = len(red.record_pos)
            action = env.scene.bf.record["Red"][red.id].get("action_index")
            low_actions[agent].append(-1 if action is None else int(action))
            start = max(starts.get(agent, end) - 1, 0)
            if decisions[agent][-1] >= 0 and end - start >= 2:
                segments[agent].append(
                    {
                        "step": step_index, "macro_mode": decisions[agent][-1],
                        "low_action": low_actions[agent][-1], "start": start, "end": end,
                    }
                )
        if terminated.get("__all__") or truncated.get("__all__"):
            break
    red_trajectories = {
        f"red_{u.id}": np.asarray(u.record_pos, dtype=float) for u in env.scene.bf.red_uav
    }
    blue_trajectories = {
        f"blue_{u.id}": np.asarray(u.record_pos, dtype=float) for u in env.scene.bf.blue_uav
    }
    positions = [value for value in list(red_trajectories.values()) + list(blue_trajectories.values()) if len(value)]
    origin = np.zeros(3)
    if positions:
        joined = np.concatenate(positions)
        origin = np.array([joined[:, 0].min(), joined[:, 1].min(), joined[:, 2].max()])
    winner, red_alive, blue_alive = env.call_back_winner()
    return {
        "red_trajectories": red_trajectories, "blue_trajectories": blue_trajectories,
        "red_segments": segments, "decision_history": decisions,
        "low_action_history": low_actions, "winner": winner,
        "red_alive": red_alive, "blue_alive": blue_alive, "origin": origin,
    }


def _plot_five_trajectories(algo, run_dir, env_config):
    try:
        from plot_hierarchical_trajectory import plot_trajectory_3d, save_episode_trace_csv

        env = DiffusionAugmentedBattleEnv(env_config)
        output_root = run_dir / "trajectory_examples"
        for index in range(5):
            rollout = _rollout_for_trajectory(algo, env, seed=index)
            episode_dir = output_root / f"episode_{index + 1:02d}_seed_{index:03d}"
            plot_trajectory_3d(rollout, episode_dir / "episode_trajectory_3d.png")
            save_episode_trace_csv(rollout, episode_dir / "episode_trace.csv")
        print(f"已生成5组三维轨迹: {output_root}")
    except Exception as exc:
        print(f"三维轨迹生成失败，训练结果仍已保留: {exc}")


def train_diffusion_hier_from_baseline(
    baseline_bundle=None,
    predictor_bundle=None,
    blue_policy_mode="script",
    blue_opponent_checkpoint=None,
    fallback_to_script=True,
    run_name_prefix="PPO_DIFFUSION",
    seed=0,
    num_cpus=8,
    num_gpus=1,
    num_env_runners=2,
    learner_gpus=0.2,
    worker_gpus=0.4,
    training_iterations=80,
    checkpoint_frequency=20,
):
    import ray

    baseline_bundle = baseline_bundle or DiffusionHRLConfig["baseline_bundle"]
    predictor_bundle = predictor_bundle or DiffusionHRLConfig["predictor_bundle"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
    tune.register_env("DiffusionAugmentedBattleEnv", env_creator)
    env_config = {
        "num_red": 3, "num_blue": 3, "candidate_enemy_count": 2,
        "sub_steps": DiffusionHRLConfig["sub_steps"], "use_script_low_level": False,
        "low_level_policy_dir": str(Path(baseline_bundle) / "policies"),
        "predictor_bundle": str(predictor_bundle), "prediction_device": "cuda",
        "prediction_steps": (10, 20, 30), "num_inference_steps": 10,
        "blue_policy_mode": blue_policy_mode,
        "blue_opponent_checkpoint": (
            None if blue_opponent_checkpoint is None else str(blue_opponent_checkpoint)
        ),
        "fallback_to_script": bool(fallback_to_script),
    }
    algo = _build_config(env_config, learner_gpus, worker_gpus, num_env_runners).build()
    migration = migrate_baseline_policy(algo, baseline_bundle)
    output_root = Path(DiffusionHRLConfig["output_dir"]) / "checkpoints"
    run_dir = output_root / f"{run_name_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.csv"
    with progress_path.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=PROGRESS_COLUMNS).writeheader()
    manifest = {
        "seed": seed, "environment": env_config, "migration": migration,
        "resources": {"learner_gpus": learner_gpus, "worker_gpus": worker_gpus},
        "training_iterations": training_iterations, "checkpoint_frequency": checkpoint_frequency,
    }
    (run_dir / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"训练输出目录: {run_dir}")
    start = time.time()
    latest = None
    try:
        bar = tqdm(range(1, int(training_iterations) + 1), desc="Diffusion-HRL PPO", dynamic_ncols=True)
        for iteration in bar:
            result = algo.train()
            _write_progress(progress_path, result, iteration)
            elapsed = time.time() - start
            eta = elapsed / iteration * (training_iterations - iteration)
            reward = float(result.get("episode_reward_mean", 0.0))
            ready = float(_nested(result, "custom_metrics/prediction_ready_ratio_mean", 0.0))
            latency = float(_nested(result, "custom_metrics/prediction_latency_ms_mean", 0.0))
            bar.set_postfix(iter=f"{iteration}/{training_iterations}", reward=f"{reward:.2f}", eta=f"{eta/3600:.1f}h")
            print(
                f"[{iteration}/{training_iterations}] reward={reward:.3f} "
                f"ready={ready:.3f} latency={latency:.1f}ms eta={eta/3600:.2f}h"
            )
            if iteration % int(checkpoint_frequency) == 0:
                latest = algo.save(str(run_dir))
                _update_plots(progress_path, run_dir)
                print(f"已保存 checkpoint: {latest}")
        if training_iterations % checkpoint_frequency:
            latest = algo.save(str(run_dir))
        _update_plots(progress_path, run_dir)
        _plot_five_trajectories(algo, run_dir, env_config)
        return str(latest)
    finally:
        algo.stop()
        ray.shutdown()


def train_diffusion_hier_vs_pi0(
    baseline_bundle=None,
    predictor_bundle=None,
    pi0_checkpoint=None,
    seed=0,
    num_cpus=8,
    num_gpus=1,
    num_env_runners=2,
    learner_gpus=0.2,
    worker_gpus=0.4,
    training_iterations=80,
    checkpoint_frequency=20,
):
    baseline_bundle = baseline_bundle or DiffusionHRLConfig["baseline_bundle"]
    pi0_checkpoint = pi0_checkpoint or str(Path(baseline_bundle) / "high_level_checkpoint")
    return train_diffusion_hier_from_baseline(
        baseline_bundle=baseline_bundle,
        predictor_bundle=predictor_bundle,
        blue_policy_mode="history_policy",
        blue_opponent_checkpoint=pi0_checkpoint,
        fallback_to_script=False,
        run_name_prefix="PPO_DIFFUSION_PI0",
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        learner_gpus=learner_gpus,
        worker_gpus=worker_gpus,
        training_iterations=training_iterations,
        checkpoint_frequency=checkpoint_frequency,
    )
