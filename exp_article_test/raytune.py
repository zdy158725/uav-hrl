import random
import time
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from tqdm.auto import tqdm

from exp_article_test.callback import AlgoCallbacks, LowLevelCallbacks
from exp_article_test.config import (
    HRLConfig,
    LowLevelConfig,
    PathConfig,
    RayCheckpointConfig,
    RayModelConfig,
    RayTrainingConfig,
)
from exp_article_test.env import BattleEnv
from exp_article_test.low_level_env import LowLevelBattleEnv

try:
    from plot_progress_metrics import plot_metrics
except ModuleNotFoundError:
    plot_metrics = None


PROGRESS_COLUMNS = [
    "training_iteration",
    "episode_reward_mean",
    "episode_reward_min",
    "episode_reward_max",
    "episode_len_mean",
    "episodes_this_iter",
    "episodes_total",
    "time_this_iter_s",
    "time_total_s",
    "custom_metrics/log_red_win_mean_mean",
    "custom_metrics/log_red_alive_mean",
    "custom_metrics/log_blue_alive_mean",
    "custom_metrics/log_attack_ratio_mean",
    "custom_metrics/log_evade_ratio_mean",
    "custom_metrics/log_low_level_kill_events_mean",
    "custom_metrics/log_low_level_death_events_mean",
    "custom_metrics/log_low_level_out_events_mean",
]


def env_creator(config):
    return BattleEnv(env_config=config)


def low_level_env_creator(config):
    return LowLevelBattleEnv(env_config=config)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    del episode, worker, kwargs
    if agent_id.startswith("red_"):
        return "red_policy"
    raise ValueError(f"Unexpected RLlib agent id: {agent_id}")


def low_level_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    del episode, worker, kwargs
    if agent_id.startswith("red_"):
        return "low_policy"
    raise ValueError(f"Unexpected RLlib agent id: {agent_id}")


def _set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_env_config(num_red, num_blue, use_script_low_level=None, low_level_policy_dir=None):
    return {
        "num_red": num_red,
        "num_blue": num_blue,
        "candidate_enemy_count": HRLConfig["candidate_enemy_count"],
        "sub_steps": HRLConfig["sub_steps"],
        "use_blue_script": True,
        "use_script_low_level": (
            HRLConfig["use_script_low_level"] if use_script_low_level is None else use_script_low_level
        ),
        "low_level_policy_dir": low_level_policy_dir or HRLConfig["low_level_policy_dir"],
        "freeze_low_level": HRLConfig["freeze_low_level"],
    }


def _build_low_level_env_config(mode, num_red, num_blue, use_blue_script=True):
    return {
        "mode": mode,
        "num_red": num_red,
        "num_blue": num_blue,
        "use_blue_script": use_blue_script,
        "threat_count": LowLevelConfig["threat_count"],
    }


def _build_ppo_config(num_gpus, num_env_runners, env_config):
    policies = {"red_policy": PolicySpec()}
    return (
        PPOConfig()
        .environment(env="BattleEnv", disable_env_checking=True, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus, num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=num_env_runners)
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
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["red_policy"],
        )
        .callbacks(callbacks_class=AlgoCallbacks)
    )


def _build_low_level_ppo_config(num_gpus, num_env_runners, env_config):
    policies = {"low_policy": PolicySpec()}
    return (
        PPOConfig()
        .environment(env="LowLevelBattleEnv", disable_env_checking=True, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus, num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=num_env_runners)
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
            policies=policies,
            policy_mapping_fn=low_level_policy_mapping_fn,
            policies_to_train=["low_policy"],
        )
        .callbacks(callbacks_class=LowLevelCallbacks)
    )


def _format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _make_run_dir(root_dir, prefix="PPO"):
    run_root = Path(root_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _progress_csv_path(run_dir):
    return Path(run_dir) / "progress.csv"


def _progress_plot_path(run_dir):
    return Path(run_dir) / "training_metrics.png"


def _init_progress_csv(progress_csv_path):
    with progress_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_COLUMNS)
        writer.writeheader()


def _append_progress_row(progress_csv_path, train_result, iteration):
    row = {}
    for column in PROGRESS_COLUMNS:
        if column == "training_iteration":
            row[column] = train_result.get(column, iteration)
        else:
            row[column] = train_result.get(column, "")
    with progress_csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_COLUMNS)
        writer.writerow(row)


def _update_progress_plot(progress_csv_path, plot_path):
    if plot_metrics is None:
        return
    try:
        plot_metrics(progress_csv_path, plot_path)
    except Exception as exc:
        print(f"绘图更新失败，已跳过: {exc}")


def _register_envs():
    tune.register_env("BattleEnv", env_creator)
    tune.register_env("LowLevelBattleEnv", low_level_env_creator)


def _train_loop(algo, run_dir, total_iterations, checkpoint_frequency, desc):
    print(f"训练输出目录: {run_dir}")
    print(f"总轮数: {total_iterations}")
    progress_csv_path = _progress_csv_path(run_dir)
    progress_plot_path = _progress_plot_path(run_dir)
    _init_progress_csv(progress_csv_path)

    start_time = time.time()
    progress_bar = tqdm(range(1, total_iterations + 1), desc=desc, dynamic_ncols=True)
    latest_checkpoint = None

    try:
        for iteration in progress_bar:
            iter_start = time.time()
            train_result = algo.train()
            elapsed = time.time() - start_time
            avg_iter_time = elapsed / iteration
            remaining_iters = total_iterations - iteration
            eta_seconds = remaining_iters * avg_iter_time
            total_estimated = elapsed + eta_seconds
            iter_time = time.time() - iter_start

            reward_mean = train_result.get("episode_reward_mean", 0.0)
            episodes_this_iter = train_result.get("episodes_this_iter", 0)
            _append_progress_row(progress_csv_path, train_result, iteration)
            progress_bar.set_postfix(
                {
                    "iter": f"{iteration}/{total_iterations}",
                    "reward": f"{reward_mean:.2f}",
                    "eps": episodes_this_iter,
                    "iter_s": f"{iter_time:.1f}",
                    "eta": _format_duration(eta_seconds),
                    "total": _format_duration(total_estimated),
                }
            )

            print(
                f"[{iteration}/{total_iterations}] "
                f"reward={reward_mean:.3f} | "
                f"episodes={episodes_this_iter} | "
                f"elapsed={_format_duration(elapsed)} | "
                f"eta={_format_duration(eta_seconds)} | "
                f"est_total={_format_duration(total_estimated)}"
            )

            if iteration % checkpoint_frequency == 0:
                latest_checkpoint = algo.save(run_dir)
                _update_progress_plot(progress_csv_path, progress_plot_path)
                print(f"已保存 checkpoint: {latest_checkpoint}")

        latest_checkpoint = algo.save(run_dir)
        _update_progress_plot(progress_csv_path, progress_plot_path)
        print(f"训练完成，最终 checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    finally:
        progress_bar.close()


def _latest_checkpoint_in_dir(run_dir):
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint_*"))
    if checkpoints:
        return checkpoints[-1]
    nested_checkpoints = sorted(run_dir.rglob("checkpoint_*"))
    return nested_checkpoints[-1] if nested_checkpoints else None


def _resolve_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir() and (checkpoint_path / "rllib_checkpoint.json").exists():
        return checkpoint_path
    if checkpoint_path.is_file():
        return checkpoint_path.parent
    return _latest_checkpoint_in_dir(checkpoint_path)


def train_hier_v1(seed, num_cpus, num_gpus, num_env_runners, use_script_low_level=None, low_level_policy_dir=None):
    import ray

    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
    _set_seed(seed)
    _register_envs()

    env_config = _build_env_config(
        num_red=HRLConfig["start_train_num_red"],
        num_blue=HRLConfig["start_train_num_blue"],
        use_script_low_level=use_script_low_level,
        low_level_policy_dir=low_level_policy_dir,
    )
    algo = _build_ppo_config(
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        env_config=env_config,
    ).build()
    run_dir = _make_run_dir(PathConfig["checkpoint_path"], prefix="PPO")

    try:
        return str(
            _train_loop(
                algo=algo,
                run_dir=run_dir,
                total_iterations=RayCheckpointConfig["training_iteration"],
                checkpoint_frequency=RayCheckpointConfig["checkpoint_frequency"],
                desc="High-Level PPO",
            )
        )
    finally:
        algo.stop()
        ray.shutdown()


def _train_low_level(mode, seed, num_cpus, num_gpus, num_env_runners):
    import ray

    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
    _set_seed(seed)
    _register_envs()

    env_config = _build_low_level_env_config(
        mode=mode,
        num_red=LowLevelConfig["start_train_num_red"],
        num_blue=LowLevelConfig["start_train_num_blue"],
        use_blue_script=True,
    )
    algo = _build_low_level_ppo_config(
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        env_config=env_config,
    ).build()
    run_dir = _make_run_dir(Path(PathConfig["low_level_path"]) / mode, prefix="PPO")

    try:
        return str(
            _train_loop(
                algo=algo,
                run_dir=run_dir,
                total_iterations=RayCheckpointConfig["training_iteration"],
                checkpoint_frequency=RayCheckpointConfig["checkpoint_frequency"],
                desc=f"Low-Level {mode.capitalize()} PPO",
            )
        )
    finally:
        algo.stop()
        ray.shutdown()


def train_low_level_attack(seed, num_cpus, num_gpus, num_env_runners):
    return _train_low_level(
        mode="attack",
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
    )


def train_low_level_evade(seed, num_cpus, num_gpus, num_env_runners):
    return _train_low_level(
        mode="evade",
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
    )


def export_low_level_policies(checkpoint_path, mode, output_dir=None, num_gpus=0):
    import ray

    if mode not in {"attack", "evade"}:
        raise ValueError(f"Unsupported low-level mode: {mode}")

    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    if resolved_checkpoint is None or not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Unable to locate checkpoint under: {checkpoint_path}")

    ray.shutdown()
    ray.init(num_cpus=1, num_gpus=num_gpus, ignore_reinit_error=True)
    _register_envs()
    env_config = _build_low_level_env_config(
        mode=mode,
        num_red=LowLevelConfig["start_train_num_red"],
        num_blue=LowLevelConfig["start_train_num_blue"],
        use_blue_script=True,
    )
    algo = _build_low_level_ppo_config(num_gpus=num_gpus, num_env_runners=0, env_config=env_config).build()

    try:
        algo.restore(str(resolved_checkpoint))
        policy = algo.get_policy("low_policy")
        model = policy.model.cpu()
        model.eval()

        export_dir = Path(output_dir or PathConfig["policy_path"])
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"low_{mode}_policy.pt"
        torch.save(model, export_path)
        print(f"已导出低层策略: {export_path}")
        return str(export_path)
    finally:
        algo.stop()
        ray.shutdown()


def smoke_test_hier_env(seed=0, num_red=1, num_blue=1, rollout_steps=3, use_script_low_level=True, low_level_policy_dir=None):
    _set_seed(seed)
    env = BattleEnv(
        env_config=_build_env_config(
            num_red=num_red,
            num_blue=num_blue,
            use_script_low_level=use_script_low_level,
            low_level_policy_dir=low_level_policy_dir,
        )
    )
    obs, info = env.reset()

    summary = {
        "initial_agents": sorted(list(obs.keys())),
        "obs_shapes": {agent_id: tuple(agent_obs.shape) for agent_id, agent_obs in obs.items()},
        "steps": [],
        "info_keys": {agent_id: sorted(agent_info.keys()) for agent_id, agent_info in info.items()},
    }

    for step_idx in range(rollout_steps):
        if not obs:
            break
        actions = {agent_id: 0 if step_idx % 2 else 1 for agent_id in obs.keys()}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        summary["steps"].append(
            {
                "step": step_idx,
                "reward_keys": sorted(list(rewards.keys())),
                "terminated_all": terminated["__all__"],
                "truncated_all": truncated["__all__"],
                "attack_ratios": {
                    agent_id: round(agent_info.get("step_attack_ratio", 0.0), 4)
                    for agent_id, agent_info in infos.items()
                },
            }
        )
        if terminated["__all__"] or truncated["__all__"]:
            break

    winner, red_alive, blue_alive = env.call_back_winner()
    summary["final"] = {
        "winner": winner,
        "red_alive": red_alive,
        "blue_alive": blue_alive,
    }
    return summary


def smoke_test_low_level_env(mode, seed=0, num_red=1, num_blue=1, rollout_steps=5):
    _set_seed(seed)
    env = LowLevelBattleEnv(
        env_config=_build_low_level_env_config(
            mode=mode,
            num_red=num_red,
            num_blue=num_blue,
            use_blue_script=True,
        )
    )
    obs, info = env.reset()

    summary = {
        "mode": mode,
        "initial_agents": sorted(list(obs.keys())),
        "obs_shapes": {agent_id: tuple(agent_obs.shape) for agent_id, agent_obs in obs.items()},
        "target_ids": {agent_id: agent_info.get("target_id") for agent_id, agent_info in info.items()},
        "steps": [],
    }

    for step_idx in range(rollout_steps):
        if not obs:
            break
        actions = {agent_id: int(step_idx % 15) for agent_id in obs.keys()}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        summary["steps"].append(
            {
                "step": step_idx,
                "reward_mean": round(float(np.mean(list(rewards.values()))) if rewards else 0.0, 4),
                "terminated_all": terminated["__all__"],
                "truncated_all": truncated["__all__"],
                "targets": {agent_id: agent_info.get("target_id") for agent_id, agent_info in infos.items()},
            }
        )
        if terminated["__all__"] or truncated["__all__"]:
            break

    winner, red_alive, blue_alive = env.call_back_winner()
    summary["final"] = {
        "winner": winner,
        "red_alive": red_alive,
        "blue_alive": blue_alive,
        "stats": env.call_back_low_level_stats(),
    }
    return summary


def smoke_test_low_level_training(mode, seed, num_cpus, num_gpus, num_env_runners, training_iterations=1):
    import ray

    ray.shutdown()
    ray.init(num_cpus=max(2, num_env_runners + 1), num_gpus=num_gpus, ignore_reinit_error=True)
    _set_seed(seed)
    _register_envs()

    env_config = _build_low_level_env_config(mode=mode, num_red=1, num_blue=1, use_blue_script=True)
    algo = _build_low_level_ppo_config(
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        env_config=env_config,
    ).build()

    results = []
    try:
        for _ in range(training_iterations):
            train_result = algo.train()
            results.append(
                {
                    "episode_reward_mean": train_result.get("episode_reward_mean"),
                    "episodes_this_iter": train_result.get("episodes_this_iter"),
                }
            )
        checkpoint_dir = Path(PathConfig["low_level_path"]) / "smoke" / mode
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = algo.save(checkpoint_dir)
        return {"mode": mode, "iterations": results, "checkpoint": str(save_path)}
    finally:
        algo.stop()
        ray.shutdown()


def smoke_test_hier_training(seed, num_cpus, num_gpus, num_env_runners, training_iterations=1):
    import ray

    del num_cpus
    ray.shutdown()
    ray.init(num_cpus=max(2, num_env_runners + 1), num_gpus=num_gpus, ignore_reinit_error=True)
    _set_seed(seed)
    _register_envs()

    env_config = _build_env_config(num_red=1, num_blue=1)
    algo = _build_ppo_config(
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        env_config=env_config,
    ).build()

    results = []
    try:
        for _ in range(training_iterations):
            train_result = algo.train()
            results.append(
                {
                    "episode_reward_mean": train_result.get("episode_reward_mean"),
                    "episodes_this_iter": train_result.get("episodes_this_iter"),
                }
            )

        checkpoint_dir = Path(PathConfig["checkpoint_path"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = algo.save(checkpoint_dir)
        return {"iterations": results, "checkpoint": str(save_path)}
    finally:
        algo.stop()
        ray.shutdown()


train = train_hier_v1
