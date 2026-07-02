from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from exp_article_test.config import HRLConfig
from exp_article_test.env import BattleEnv
from exp_article_test.raytune import _resolve_checkpoint_path


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CN_FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
POLICY_ID = "red_policy"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_policy(checkpoint: Path):
    from ray.rllib.policy.policy import Policy

    resolved = _resolve_checkpoint_path(checkpoint)
    if resolved is None:
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}")
    loaded = Policy.from_checkpoint(str(resolved), policy_ids=[POLICY_ID])
    return loaded[POLICY_ID] if isinstance(loaded, dict) else loaded


def _checkpoint_iteration(checkpoint: Path) -> int:
    match = re.search(r"checkpoint_(\d+)", checkpoint.name)
    return int(match.group(1)) if match else -1


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _build_eval_env(
    opponent_mode: str,
    baseline_checkpoint: Path | None,
    num_red: int,
    num_blue: int,
    use_script_low_level: bool,
    low_level_policy_dir: str | None,
) -> BattleEnv:
    env_config = {
        "num_red": num_red,
        "num_blue": num_blue,
        "candidate_enemy_count": HRLConfig["candidate_enemy_count"],
        "sub_steps": HRLConfig["sub_steps"],
        "use_blue_script": True,
        "use_script_low_level": use_script_low_level,
        "low_level_policy_dir": low_level_policy_dir or HRLConfig["low_level_policy_dir"],
        "freeze_low_level": True,
        "blue_policy_mode": "script" if opponent_mode == "script" else "history_policy",
        "blue_opponent_checkpoint": (
            None if opponent_mode == "script" else str(baseline_checkpoint)
        ),
        "fallback_to_script": opponent_mode == "script",
    }
    return BattleEnv(env_config=env_config)


def _run_episode(env: BattleEnv, policy, seed: int, max_steps: int) -> dict:
    _set_seed(seed)
    observations, _ = env.reset(seed=seed)
    reward_total = 0.0
    attack_count = 0
    evade_count = 0
    episode_steps = 0

    for episode_steps in range(1, max_steps + 1):
        actions = {}
        for agent_id, observation in observations.items():
            result = policy.compute_single_action(
                np.asarray(observation, dtype=np.float32),
                explore=False,
            )
            action = result[0] if isinstance(result, tuple) else result
            action = int(action)
            actions[agent_id] = action
            if action <= 0:
                evade_count += 1
            else:
                attack_count += 1

        observations, rewards, terminated, truncated, _ = env.step(actions)
        reward_total += float(sum(rewards.values()))
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            break

    winner, red_alive, blue_alive = env.call_back_winner()
    decision_count = attack_count + evade_count
    return {
        "seed": seed,
        "winner": winner,
        "red_win": int(winner == "red"),
        "blue_win": int(winner == "blue"),
        "draw": int(winner not in {"red", "blue"}),
        "episode_reward": reward_total,
        "episode_length": episode_steps,
        "red_alive": red_alive,
        "blue_alive": blue_alive,
        "attack_ratio": attack_count / decision_count if decision_count else 0.0,
        "evade_ratio": evade_count / decision_count if decision_count else 0.0,
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _summarize(checkpoint: Path, opponent_mode: str, rows: list[dict]) -> dict:
    episodes = len(rows)
    red_wins = sum(int(row["red_win"]) for row in rows)
    blue_wins = sum(int(row["blue_win"]) for row in rows)
    decisive_episodes = red_wins + blue_wins
    low, high = _wilson_interval(red_wins, episodes)
    no_draw_low, no_draw_high = _wilson_interval(red_wins, decisive_episodes)
    return {
        "checkpoint": str(checkpoint),
        "training_iteration": _checkpoint_iteration(checkpoint),
        "opponent": opponent_mode,
        "episodes": episodes,
        "red_win_rate": red_wins / episodes if episodes else 0.0,
        "red_win_ci95_low": low,
        "red_win_ci95_high": high,
        "decisive_episodes": decisive_episodes,
        "red_win_rate_no_draw": red_wins / decisive_episodes if decisive_episodes else 0.0,
        "red_win_no_draw_ci95_low": no_draw_low,
        "red_win_no_draw_ci95_high": no_draw_high,
        "blue_win_rate": _mean(rows, "blue_win"),
        "draw_rate": _mean(rows, "draw"),
        "episode_reward_mean": _mean(rows, "episode_reward"),
        "episode_len_mean": _mean(rows, "episode_length"),
        "red_alive_mean": _mean(rows, "red_alive"),
        "blue_alive_mean": _mean(rows, "blue_alive"),
        "attack_ratio_mean": _mean(rows, "attack_ratio"),
        "evade_ratio_mean": _mean(rows, "evade_ratio"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _annotate_bars(axis, bars, fmt="{:.3f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _plot_single_checkpoint_summary(rows: list[dict], output_path: Path) -> None:
    opponent_labels = {"script": "增强脚本蓝方", "initial_policy": "初始历史策略 π0"}
    ordered = [
        next((row for row in rows if row["opponent"] == opponent), None)
        for opponent in ("script", "initial_policy")
    ]
    ordered = [row for row in ordered if row is not None]
    labels = [opponent_labels[row["opponent"]] for row in ordered]
    x = np.arange(len(ordered), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    raw_win = np.asarray([row["red_win_rate"] for row in ordered], dtype=float)
    no_draw_win = np.asarray([row["red_win_rate_no_draw"] for row in ordered], dtype=float)
    raw_error = np.asarray(
        [
            raw_win - np.asarray([row["red_win_ci95_low"] for row in ordered], dtype=float),
            np.asarray([row["red_win_ci95_high"] for row in ordered], dtype=float) - raw_win,
        ]
    )
    no_draw_error = np.asarray(
        [
            no_draw_win
            - np.asarray([row["red_win_no_draw_ci95_low"] for row in ordered], dtype=float),
            np.asarray([row["red_win_no_draw_ci95_high"] for row in ordered], dtype=float)
            - no_draw_win,
        ]
    )
    raw_bars = axes[0].bar(
        x - width / 2,
        raw_win,
        width,
        yerr=raw_error,
        capsize=4,
        color="#7a8793",
        label="含平局",
    )
    no_draw_bars = axes[0].bar(
        x + width / 2,
        no_draw_win,
        width,
        yerr=no_draw_error,
        capsize=4,
        color="#b33b2e",
        label="不含平局",
    )
    _annotate_bars(axes[0], raw_bars)
    _annotate_bars(axes[0], no_draw_bars)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_title("固定对手胜率", fontproperties=CN_FONT, fontsize=13)
    axes[0].set_ylabel("红方胜率", fontproperties=CN_FONT)
    axes[0].legend(prop=CN_FONT)

    reward_bars = axes[1].bar(
        x,
        [row["episode_reward_mean"] for row in ordered],
        width=0.55,
        color=["#a33d32" if row["opponent"] == "script" else "#245b78" for row in ordered],
    )
    _annotate_bars(axes[1], reward_bars, fmt="{:.2f}")
    axes[1].set_title("平均回合奖励", fontproperties=CN_FONT, fontsize=13)
    axes[1].set_ylabel("奖励", fontproperties=CN_FONT)

    red_bars = axes[2].bar(
        x - width / 2,
        [row["red_alive_mean"] for row in ordered],
        width,
        color="#b33b2e",
        label="红方存活",
    )
    blue_bars = axes[2].bar(
        x + width / 2,
        [row["blue_alive_mean"] for row in ordered],
        width,
        color="#245b78",
        label="蓝方存活",
    )
    _annotate_bars(axes[2], red_bars)
    _annotate_bars(axes[2], blue_bars)
    axes[2].set_title("双方平均存活数", fontproperties=CN_FONT, fontsize=13)
    axes[2].set_ylabel("存活无人机数量", fontproperties=CN_FONT)
    axes[2].legend(prop=CN_FONT)

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontproperties=CN_FONT)
        axis.grid(axis="y", alpha=0.25)
        for label in axis.get_yticklabels():
            if CN_FONT is not None:
                label.set_fontproperties(CN_FONT)

    iteration = int(ordered[0]["training_iteration"])
    fig.suptitle(
        f"Checkpoint {iteration} 固定基准评估",
        fontproperties=CN_FONT,
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(rows: list[dict], output_path: Path) -> None:
    iterations = {int(row["training_iteration"]) for row in rows}
    if len(iterations) == 1:
        _plot_single_checkpoint_summary(rows, output_path)
        return

    labels = {"script": "增强脚本蓝方", "initial_policy": "初始历史策略 π0"}
    colors = {"script": "#a33d32", "initial_policy": "#245b78"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for opponent in ("script", "initial_policy"):
        selected = sorted(
            (row for row in rows if row["opponent"] == opponent),
            key=lambda row: int(row["training_iteration"]),
        )
        if not selected:
            continue
        x = np.asarray([row["training_iteration"] for row in selected], dtype=int)
        win = np.asarray([row["red_win_rate"] for row in selected], dtype=float)
        low = np.asarray([row["red_win_ci95_low"] for row in selected], dtype=float)
        high = np.asarray([row["red_win_ci95_high"] for row in selected], dtype=float)
        no_draw_win = np.asarray([row["red_win_rate_no_draw"] for row in selected], dtype=float)
        no_draw_low = np.asarray([row["red_win_no_draw_ci95_low"] for row in selected], dtype=float)
        no_draw_high = np.asarray([row["red_win_no_draw_ci95_high"] for row in selected], dtype=float)
        axes[0].plot(
            x,
            win,
            marker="o",
            linewidth=2.2,
            color=colors[opponent],
            label=f"{labels[opponent]}（含平局）",
        )
        axes[0].fill_between(x, low, high, color=colors[opponent], alpha=0.15)
        axes[0].plot(
            x,
            no_draw_win,
            marker="D",
            linestyle="--",
            linewidth=1.9,
            color=colors[opponent],
            label=f"{labels[opponent]}（不含平局）",
        )
        axes[0].fill_between(x, no_draw_low, no_draw_high, color=colors[opponent], alpha=0.08)
        axes[1].plot(
            x,
            [row["episode_reward_mean"] for row in selected],
            marker="o",
            linewidth=2.2,
            color=colors[opponent],
            label=labels[opponent],
        )
        axes[2].plot(
            x,
            [row["red_alive_mean"] for row in selected],
            marker="o",
            linewidth=2.2,
            color=colors[opponent],
            label=f"红方存活/{labels[opponent]}",
        )
        axes[2].plot(
            x,
            [row["blue_alive_mean"] for row in selected],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color=colors[opponent],
            alpha=0.75,
            label=f"蓝方存活/{labels[opponent]}",
        )

    titles = ("固定对手胜率（含/不含平局）", "平均回合奖励", "双方平均存活数")
    ylabels = ("红方胜率", "奖励", "存活无人机数量")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title, fontproperties=CN_FONT, fontsize=13)
        axis.set_xlabel("训练轮次", fontproperties=CN_FONT)
        axis.set_ylabel(ylabel, fontproperties=CN_FONT)
        axis.grid(alpha=0.25)
        axis.legend(prop=CN_FONT, fontsize=9)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            if CN_FONT is not None:
                label.set_fontproperties(CN_FONT)
    axes[0].set_ylim(0.0, 1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    checkpoints = [path.resolve() for path in args.checkpoints]
    if not checkpoints:
        checkpoints = sorted(run_dir.glob("checkpoint_*"), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"目录中没有 checkpoint: {run_dir}")

    baseline_checkpoint = args.baseline_checkpoint.resolve()
    if _resolve_checkpoint_path(baseline_checkpoint) is None:
        raise FileNotFoundError(f"找不到初始历史策略: {baseline_checkpoint}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + index for index in range(args.episodes)]
    episode_rows = []
    summary_rows = []

    for checkpoint in checkpoints:
        policy = _load_policy(checkpoint)
        for opponent_mode in args.opponents:
            env = _build_eval_env(
                opponent_mode=opponent_mode,
                baseline_checkpoint=baseline_checkpoint,
                num_red=args.num_red,
                num_blue=args.num_blue,
                use_script_low_level=not args.use_trained_low_level,
                low_level_policy_dir=args.low_level_policy_dir,
            )
            rows = []
            for episode_index, seed in enumerate(seeds, start=1):
                result = _run_episode(env, policy, seed=seed, max_steps=args.max_steps)
                result.update(
                    {
                        "checkpoint": str(checkpoint),
                        "training_iteration": _checkpoint_iteration(checkpoint),
                        "opponent": opponent_mode,
                        "episode": episode_index,
                    }
                )
                rows.append(result)
                if episode_index % max(1, min(50, args.episodes)) == 0 or episode_index == args.episodes:
                    print(
                        f"checkpoint={checkpoint.name} opponent={opponent_mode} "
                        f"episode={episode_index}/{args.episodes} "
                        f"win_rate={_mean(rows, 'red_win'):.3f}"
                    )
            episode_rows.extend(rows)
            summary_rows.append(_summarize(checkpoint, opponent_mode, rows))

    _write_csv(output_dir / "fixed_benchmark_episodes.csv", episode_rows)
    _write_csv(output_dir / "fixed_benchmark_summary.csv", summary_rows)
    _plot_summary(summary_rows, output_dir / "fixed_benchmark_comparison.png")
    with (output_dir / "evaluation_config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2, default=str)
    print(f"评估完成，结果目录: {output_dir}")
    return output_dir


def evaluate_fixed_checkpoints(
    run_dir: Path,
    baseline_checkpoint: Path,
    checkpoints: list[Path],
    output_dir: Path,
    episodes: int = 200,
    max_steps: int = 120,
    seed: int = 20260625,
    num_red: int = 3,
    num_blue: int = 3,
    opponents: tuple[str, ...] = ("script", "initial_policy"),
    use_trained_low_level: bool = False,
    low_level_policy_dir: str | None = None,
) -> Path:
    """Programmatic entry point used by the high-level training loop."""
    args = argparse.Namespace(
        run_dir=Path(run_dir),
        baseline_checkpoint=Path(baseline_checkpoint),
        checkpoints=[Path(path) for path in checkpoints],
        opponents=tuple(opponents),
        episodes=int(episodes),
        max_steps=int(max_steps),
        seed=int(seed),
        num_red=int(num_red),
        num_blue=int(num_blue),
        use_trained_low_level=bool(use_trained_low_level),
        low_level_policy_dir=low_level_policy_dir,
        output_dir=Path(output_dir),
    )
    return evaluate(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定对手评估多个高层 PPO checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--opponents",
        choices=("script", "initial_policy"),
        nargs="+",
        default=("script", "initial_policy"),
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--num-red", type=int, default=3)
    parser.add_argument("--num-blue", type=int, default=3)
    parser.add_argument("--use-trained-low-level", action="store_true")
    parser.add_argument("--low-level-policy-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "evaluation" / "fixed_self_play_benchmark",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
