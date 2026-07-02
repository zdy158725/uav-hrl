from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
sys.path.insert(0, str(BASE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from exp_article_test.env import BattleEnv
from exp_article_test.raytune import _build_env_config, _build_ppo_config, _register_envs, _resolve_checkpoint_path


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("checkpoint_*"))
    return candidates[-1] if candidates else None


def _load_algo(checkpoint_path: Path, env_config: dict, num_gpus: int = 0):
    import ray

    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(f"Cannot find checkpoint under: {checkpoint_path}")

    ray.shutdown()
    ray.init(num_cpus=1, num_gpus=num_gpus, ignore_reinit_error=True, include_dashboard=False)
    _register_envs()
    algo = _build_ppo_config(num_gpus=num_gpus, num_env_runners=0, env_config=env_config).build()
    algo.restore(str(resolved))
    return ray, algo


def _heuristic_action(obs: np.ndarray) -> int:
    # For no-checkpoint visualization: attack the nearest candidate if it exists.
    first_enemy_dist_idx = 8 + 6
    if obs.shape[0] > first_enemy_dist_idx and obs[first_enemy_dist_idx] > 0.0:
        return 1
    return 0


def rollout_episode(
    num_red: int,
    num_blue: int,
    max_steps: int,
    seed: int,
    checkpoint: Path | None = None,
    use_script_low_level: bool = True,
    low_level_policy_dir: str | None = None,
    num_gpus: int = 0,
) -> dict:
    _set_seed(seed)
    env_config = _build_env_config(
        num_red=num_red,
        num_blue=num_blue,
        use_script_low_level=use_script_low_level,
        low_level_policy_dir=low_level_policy_dir,
    )
    env = BattleEnv(env_config=env_config)

    ray = None
    algo = None
    if checkpoint is not None:
        ray, algo = _load_algo(checkpoint, env_config=env_config, num_gpus=num_gpus)

    try:
        obs, _info = env.reset()
        agents = [f"red_{idx}" for idx in range(num_red)]
        decision_history = {agent_id: [] for agent_id in agents}
        low_action_history = {agent_id: [] for agent_id in agents}

        for _step_idx in range(max_steps):
            if not obs:
                break
            action_dict = {}
            for agent_id, agent_obs in obs.items():
                if algo is None:
                    action = _heuristic_action(agent_obs)
                else:
                    action = algo.compute_single_action(agent_obs, policy_id="red_policy", explore=False)
                    if isinstance(action, tuple):
                        action = action[0]
                action_dict[agent_id] = int(action)

            for agent_id in agents:
                action = action_dict.get(agent_id)
                if action is None:
                    decision_history[agent_id].append(-1)
                else:
                    decision_history[agent_id].append(0 if int(action) <= 0 else 1)

            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)

            for agent_id in agents:
                red_idx = int(agent_id.split("_")[1])
                if red_idx < len(env.scene.bf.red_uav):
                    red_id = env.scene.bf.red_uav[red_idx].id
                    record = env.scene.bf.record["Red"].get(red_id, {})
                    low_action = record.get("action_index")
                    low_action_history[agent_id].append(-1 if low_action is None else int(low_action))

            if terminated["__all__"] or truncated["__all__"]:
                break

        winner, red_alive, blue_alive = env.call_back_winner()
        return {
            "decision_history": decision_history,
            "low_action_history": low_action_history,
            "winner": winner,
            "red_alive": red_alive,
            "blue_alive": blue_alive,
        }
    finally:
        if algo is not None:
            algo.stop()
        if ray is not None:
            ray.shutdown()


def plot_episode_decisions(rollout: dict, output_path: Path, dt: float = 1.0) -> Path:
    decision_history = rollout["decision_history"]
    agents = list(decision_history.keys())
    max_len = max((len(values) for values in decision_history.values()), default=0)
    if max_len == 0:
        raise ValueError("No decision history to plot.")

    matrix = np.full((len(agents), max_len), -1, dtype=float)
    for row_idx, agent_id in enumerate(agents):
        values = decision_history[agent_id]
        matrix[row_idx, : len(values)] = values

    masked_matrix = np.ma.masked_where(matrix < 0, matrix)
    cmap = ListedColormap(["#1f77b4", "#d62728"])
    cmap.set_bad("#e5e5e5")

    fig_height = max(3.2, 0.55 * len(agents) + 2.4)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.imshow(masked_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ax.set_title(
        "Episode-Level High-Level Decisions\n"
        f"winner={rollout['winner']}, red_alive={rollout['red_alive']}, blue_alive={rollout['blue_alive']}"
    )
    ax.set_xlabel("High-Level Decision Step")
    ax.set_ylabel("Red UAV")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)

    tick_count = min(12, max_len)
    tick_indices = np.linspace(0, max_len - 1, tick_count, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{idx * dt:.0f}" for idx in tick_indices])
    ax.set_xlabel("High-Level Decision Time / step")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Attack"),
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="Evade"),
        plt.Rectangle((0, 0), 1, 1, color="#e5e5e5", label="Inactive / dead"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncols=3, frameon=True)
    ax.grid(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot one episode's high-level attack/evade decision timeline.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional high-level PPO checkpoint or run dir.")
    parser.add_argument("--num-red", type=int, default=2)
    parser.add_argument("--num-blue", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--use-trained-low-level", action="store_true", help="Load exported low-level policies.")
    parser.add_argument("--low-level-policy-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "outputs" / "figures" / "episode_high_level_decisions.png",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is not None and checkpoint.is_dir() and not (checkpoint / "rllib_checkpoint.json").exists():
        checkpoint = _latest_checkpoint(checkpoint)

    rollout = rollout_episode(
        num_red=args.num_red,
        num_blue=args.num_blue,
        max_steps=args.max_steps,
        seed=args.seed,
        checkpoint=checkpoint,
        use_script_low_level=not args.use_trained_low_level,
        low_level_policy_dir=args.low_level_policy_dir,
        num_gpus=args.num_gpus,
    )
    saved_path = plot_episode_decisions(rollout, args.output)
    print(saved_path)


if __name__ == "__main__":
    main()
