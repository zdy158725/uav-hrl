from __future__ import annotations

import argparse
import csv
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
from matplotlib.font_manager import FontProperties

from exp_article_test.env import BattleEnv
from exp_article_test.raytune import _build_env_config, _build_ppo_config, _register_envs, _resolve_checkpoint_path


ATTACK_COLOR = "#d62728"
EVADE_COLOR = "#1f77b4"
BLUE_COLOR = "#4c566a"
INACTIVE_COLOR = "#e5e5e5"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CN_FONT = FontProperties(fname=FONT_PATH)
CN_BOLD_FONT = FontProperties(fname=BOLD_FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False


def _apply_cn_axis_font(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)
    if hasattr(ax, "get_zticklabels"):
        for label in ax.get_zticklabels():
            label.set_fontproperties(CN_FONT)


def _format_agent_label(agent_id: str) -> str:
    prefix, idx = agent_id.split("_")
    return f"{'红方' if prefix == 'red' else '蓝方'}{idx}"


def _winner_text(winner: str) -> str:
    return {"red": "红方胜利", "blue": "蓝方胜利", "draw": "平局"}.get(winner, winner)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("checkpoint_*"))
    return candidates[-1] if candidates else None


def _resolve_optional_checkpoint(checkpoint: Path | None) -> Path | None:
    if checkpoint is None:
        return None
    if checkpoint.is_dir() and not (checkpoint / "rllib_checkpoint.json").exists():
        return _latest_checkpoint(checkpoint)
    return checkpoint


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
    # For visualization without a trained checkpoint: attack nearest visible enemy.
    first_enemy_dist_idx = 8 + 6
    if obs.shape[0] > first_enemy_dist_idx and obs[first_enemy_dist_idx] > 0.0:
        return 1
    return 0


def _relative_xyz(pos: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = pos[:, 0] - origin[0]
    y = pos[:, 1] - origin[1]
    h = (-pos[:, 2]) - (-origin[2])
    return x, y, h


def _macro_label(action: int | None) -> int:
    if action is None:
        return -1
    return 0 if int(action) <= 0 else 1


def rollout_hierarchical_episode(
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
    checkpoint = _resolve_optional_checkpoint(checkpoint)
    if checkpoint is not None:
        ray, algo = _load_algo(checkpoint, env_config=env_config, num_gpus=num_gpus)

    try:
        obs, _info = env.reset()
        red_agents = [f"red_{idx}" for idx in range(num_red)]
        decision_history = {agent_id: [] for agent_id in red_agents}
        low_action_history = {agent_id: [] for agent_id in red_agents}
        red_segments = {agent_id: [] for agent_id in red_agents}

        for step_idx in range(max_steps):
            if not obs:
                break

            start_lengths = {
                f"red_{red.id}": len(red.record_pos)
                for red in env.scene.bf.red_uav
            }
            action_dict = {}
            for agent_id, agent_obs in obs.items():
                if algo is None:
                    action = _heuristic_action(agent_obs)
                else:
                    action = algo.compute_single_action(agent_obs, policy_id="red_policy", explore=False)
                    if isinstance(action, tuple):
                        action = action[0]
                action_dict[agent_id] = int(action)

            for agent_id in red_agents:
                decision_history[agent_id].append(_macro_label(action_dict.get(agent_id)))

            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)

            for red in env.scene.bf.red_uav:
                agent_id = f"red_{red.id}"
                start_len = start_lengths.get(agent_id, len(red.record_pos))
                end_len = len(red.record_pos)
                record = env.scene.bf.record["Red"].get(red.id, {})
                low_action = record.get("action_index")
                low_action_history[agent_id].append(-1 if low_action is None else int(low_action))

                macro_mode = decision_history[agent_id][-1]
                if macro_mode < 0 or end_len <= 1:
                    continue

                # Include the previous point so each colored segment connects to the prior segment.
                seg_start = max(start_len - 1, 0)
                if end_len - seg_start >= 2:
                    red_segments[agent_id].append(
                        {
                            "step": step_idx,
                            "macro_mode": macro_mode,
                            "low_action": -1 if low_action is None else int(low_action),
                            "start": seg_start,
                            "end": end_len,
                        }
                    )

            if terminated["__all__"] or truncated["__all__"]:
                break

        red_trajectories = {
            f"red_{red.id}": np.asarray(red.record_pos, dtype=float)
            for red in env.scene.bf.red_uav
        }
        blue_trajectories = {
            f"blue_{blue.id}": np.asarray(blue.record_pos, dtype=float)
            for blue in env.scene.bf.blue_uav
        }
        all_positions = [
            pos
            for pos in list(red_trajectories.values()) + list(blue_trajectories.values())
            if len(pos) > 0
        ]
        if all_positions:
            all_pos = np.concatenate(all_positions, axis=0)
            origin = np.array([np.min(all_pos[:, 0]), np.min(all_pos[:, 1]), np.max(all_pos[:, 2])])
        else:
            origin = np.zeros(3)
        winner, red_alive, blue_alive = env.call_back_winner()

        return {
            "red_trajectories": red_trajectories,
            "blue_trajectories": blue_trajectories,
            "red_segments": red_segments,
            "decision_history": decision_history,
            "low_action_history": low_action_history,
            "winner": winner,
            "red_alive": red_alive,
            "blue_alive": blue_alive,
            "origin": origin,
        }
    finally:
        if algo is not None:
            algo.stop()
        if ray is not None:
            ray.shutdown()


def _set_equal_3d_axes(ax, xs, ys, hs):
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    h_min, h_max = np.min(hs), np.max(hs)
    max_range = max(x_max - x_min, y_max - y_min, h_max - h_min, 1.0)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    h_mid = 0.5 * (h_min + h_max)
    half = max_range * 0.55

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(h_mid - half, h_mid + half)


def _plot_start_end_2d(ax, x: np.ndarray, y: np.ndarray):
    if len(x) == 0:
        return
    ax.scatter(x[0], y[0], marker="o", s=38, color="white", edgecolor="black", linewidth=1.0, zorder=5)
    ax.scatter(x[-1], y[-1], marker="x", s=55, color="black", linewidth=1.4, zorder=6)


def _plot_start_end_3d(ax, x: np.ndarray, y: np.ndarray, h: np.ndarray):
    if len(x) == 0:
        return
    ax.scatter(x[0], y[0], h[0], marker="o", s=34, color="white", edgecolor="black", linewidth=1.0)
    ax.scatter(x[-1], y[-1], h[-1], marker="x", s=50, color="black", linewidth=1.4)


def _plot_red_segments_2d(ax, rollout: dict):
    origin = rollout["origin"]
    for agent_id, pos in rollout["red_trajectories"].items():
        if len(pos) < 2:
            continue
        x, y, _h = _relative_xyz(pos, origin)
        for seg in rollout["red_segments"].get(agent_id, []):
            color = ATTACK_COLOR if seg["macro_mode"] == 1 else EVADE_COLOR
            s, e = seg["start"], seg["end"]
            ax.plot(x[s:e], y[s:e], color=color, linewidth=2.2, alpha=0.9)
        _plot_start_end_2d(ax, x, y)
        ax.text(x[0], y[0], f" {_format_agent_label(agent_id)}", fontproperties=CN_FONT, fontsize=8, color=ATTACK_COLOR)


def _plot_red_segments_3d(ax, rollout: dict):
    origin = rollout["origin"]
    for agent_id, pos in rollout["red_trajectories"].items():
        if len(pos) < 2:
            continue
        x, y, h = _relative_xyz(pos, origin)
        for seg in rollout["red_segments"].get(agent_id, []):
            color = ATTACK_COLOR if seg["macro_mode"] == 1 else EVADE_COLOR
            s, e = seg["start"], seg["end"]
            ax.plot(x[s:e], y[s:e], h[s:e], color=color, linewidth=2.0, alpha=0.9)
        _plot_start_end_3d(ax, x, y, h)
        ax.text(x[0], y[0], h[0], f" {_format_agent_label(agent_id)}", fontproperties=CN_FONT, fontsize=8, color=ATTACK_COLOR)


def _plot_blue_2d(ax, rollout: dict):
    origin = rollout["origin"]
    for agent_id, pos in rollout["blue_trajectories"].items():
        if len(pos) < 2:
            continue
        x, y, _h = _relative_xyz(pos, origin)
        ax.plot(x, y, linestyle="--", color=BLUE_COLOR, linewidth=1.8, alpha=0.75)
        _plot_start_end_2d(ax, x, y)
        ax.text(x[0], y[0], f" {_format_agent_label(agent_id)}", fontproperties=CN_FONT, fontsize=8, color=BLUE_COLOR)


def _plot_blue_3d(ax, rollout: dict):
    origin = rollout["origin"]
    for agent_id, pos in rollout["blue_trajectories"].items():
        if len(pos) < 2:
            continue
        x, y, h = _relative_xyz(pos, origin)
        ax.plot(x, y, h, linestyle="--", color=BLUE_COLOR, linewidth=1.6, alpha=0.75)
        _plot_start_end_3d(ax, x, y, h)
        ax.text(x[0], y[0], h[0], f" {_format_agent_label(agent_id)}", fontproperties=CN_FONT, fontsize=8, color=BLUE_COLOR)


def _legend_handles():
    return [
        plt.Line2D([0], [0], color=ATTACK_COLOR, linewidth=2.4, label="红方进攻"),
        plt.Line2D([0], [0], color=EVADE_COLOR, linewidth=2.4, label="红方逃逸"),
        plt.Line2D([0], [0], color=BLUE_COLOR, linewidth=2.0, linestyle="--", label="蓝方脚本"),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linewidth=0, label="起点"),
        plt.Line2D([0], [0], marker="x", color="black", linewidth=0, label="终点"),
    ]


def _apply_legend_font(legend):
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)


def plot_trajectory_2d(rollout: dict, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_blue_2d(ax, rollout)
    _plot_red_segments_2d(ax, rollout)
    ax.set_title(
        "分层执行轨迹图（二维）\n"
        f"结果：{_winner_text(rollout['winner'])}，红方存活：{rollout['red_alive']}，蓝方存活：{rollout['blue_alive']}",
        fontproperties=CN_BOLD_FONT,
        fontsize=15,
    )
    ax.set_xlabel("相对 x / m", fontproperties=CN_FONT)
    ax.set_ylabel("相对 y / m", fontproperties=CN_FONT)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    _apply_cn_axis_font(ax)
    _apply_legend_font(ax.legend(handles=_legend_handles(), loc="best", frameon=True))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_trajectory_3d(rollout: dict, output_path: Path) -> Path:
    fig = plt.figure(figsize=(12.5, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _plot_blue_3d(ax, rollout)
    _plot_red_segments_3d(ax, rollout)

    all_xyz = []
    for pos in list(rollout["red_trajectories"].values()) + list(rollout["blue_trajectories"].values()):
        if len(pos) > 0:
            all_xyz.append(_relative_xyz(pos, rollout["origin"]))
    if all_xyz:
        xs = np.concatenate([item[0] for item in all_xyz])
        ys = np.concatenate([item[1] for item in all_xyz])
        hs = np.concatenate([item[2] for item in all_xyz])
        _set_equal_3d_axes(ax, xs, ys, hs)

    ax.set_title(
        "分层执行轨迹图（三维）\n"
        f"结果：{_winner_text(rollout['winner'])}，红方存活：{rollout['red_alive']}，蓝方存活：{rollout['blue_alive']}",
        fontproperties=CN_BOLD_FONT,
        fontsize=15,
    )
    ax.set_xlabel("相对 x / m", fontproperties=CN_FONT)
    ax.set_ylabel("相对 y / m", fontproperties=CN_FONT)
    ax.set_zlabel("相对高度 h=-z / m", fontproperties=CN_FONT)
    ax.view_init(elev=25, azim=-55)
    ax.grid(True, alpha=0.3)
    _apply_cn_axis_font(ax)
    _apply_legend_font(ax.legend(handles=_legend_handles(), loc="upper left", frameon=True))
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_decision_timeline(rollout: dict, output_path: Path) -> Path:
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
    cmap = ListedColormap([EVADE_COLOR, ATTACK_COLOR])
    cmap.set_bad(INACTIVE_COLOR)

    fig_height = max(3.2, 0.55 * len(agents) + 2.4)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.imshow(masked_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("单局高层决策时间序列", fontproperties=CN_BOLD_FONT, fontsize=15)
    ax.set_xlabel("高层决策步", fontproperties=CN_FONT)
    ax.set_ylabel("红方无人机", fontproperties=CN_FONT)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([_format_agent_label(agent_id) for agent_id in agents], fontproperties=CN_FONT)

    tick_count = min(12, max_len)
    tick_indices = np.linspace(0, max_len - 1, tick_count, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(idx) for idx in tick_indices], fontproperties=CN_FONT)
    legend = ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=ATTACK_COLOR, label="进攻"),
            plt.Rectangle((0, 0), 1, 1, color=EVADE_COLOR, label="逃逸"),
            plt.Rectangle((0, 0), 1, 1, color=INACTIVE_COLOR, label="不活跃/已损毁"),
        ],
        loc="upper right",
        ncols=3,
        frameon=True,
    )
    _apply_legend_font(legend)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_episode_trace_csv(rollout: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team",
                "uav_id",
                "point_index",
                "high_step",
                "macro_mode",
                "low_action",
                "x",
                "y",
                "h",
            ],
        )
        writer.writeheader()
        origin = rollout["origin"]
        for agent_id, pos in rollout["red_trajectories"].items():
            if len(pos) == 0:
                continue
            x, y, h = _relative_xyz(pos, origin)
            segment_by_point = {}
            for seg in rollout["red_segments"].get(agent_id, []):
                for point_idx in range(seg["start"], seg["end"]):
                    segment_by_point[point_idx] = seg
            for point_idx in range(len(pos)):
                seg = segment_by_point.get(point_idx, {})
                writer.writerow(
                    {
                        "team": "red",
                        "uav_id": agent_id,
                        "point_index": point_idx,
                        "high_step": seg.get("step", ""),
                        "macro_mode": seg.get("macro_mode", ""),
                        "low_action": seg.get("low_action", ""),
                        "x": x[point_idx],
                        "y": y[point_idx],
                        "h": h[point_idx],
                    }
                )
        for agent_id, pos in rollout["blue_trajectories"].items():
            if len(pos) == 0:
                continue
            x, y, h = _relative_xyz(pos, origin)
            for point_idx in range(len(pos)):
                writer.writerow(
                    {
                        "team": "blue",
                        "uav_id": agent_id,
                        "point_index": point_idx,
                        "high_step": "",
                        "macro_mode": "",
                        "low_action": "",
                        "x": x[point_idx],
                        "y": y[point_idx],
                        "h": h[point_idx],
                    }
                )
    return output_path


def plot_hierarchical_episode(rollout: dict, output_dir: Path, save_csv: bool = True) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        plot_trajectory_2d(rollout, output_dir / "episode_trajectory_2d.png"),
        plot_trajectory_3d(rollout, output_dir / "episode_trajectory_3d.png"),
        plot_decision_timeline(rollout, output_dir / "episode_decision_timeline.png"),
    ]
    if save_csv:
        saved_paths.append(save_episode_trace_csv(rollout, output_dir / "episode_trace.csv"))
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Plot hierarchical attack/evade trajectories for one air-combat episode.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional high-level PPO checkpoint or run dir.")
    parser.add_argument("--num-red", type=int, default=2)
    parser.add_argument("--num-blue", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--use-trained-low-level", action="store_true", help="Load exported low-level policies.")
    parser.add_argument("--low-level-policy-dir", type=str, default=None)
    parser.add_argument("--no-csv", action="store_true", help="Do not save episode_trace.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "figures" / "hierarchical_episode",
    )
    args = parser.parse_args()

    rollout = rollout_hierarchical_episode(
        num_red=args.num_red,
        num_blue=args.num_blue,
        max_steps=args.max_steps,
        seed=args.seed,
        checkpoint=args.checkpoint,
        use_script_low_level=not args.use_trained_low_level,
        low_level_policy_dir=args.low_level_policy_dir,
        num_gpus=args.num_gpus,
    )
    for saved_path in plot_hierarchical_episode(rollout, args.output_dir, save_csv=not args.no_csv):
        print(saved_path)


if __name__ == "__main__":
    main()
