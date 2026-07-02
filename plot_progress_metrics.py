from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MPLCONFIGDIR = BASE_DIR / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("episode_reward_mean", "Episode Reward Mean"),
    ("custom_metrics/log_red_win_mean_mean", "Red Win Rate (Including Draws)"),
    ("custom_metrics/log_red_win_no_draw_mean_mean", "Red Win Rate (Excluding Draws)"),
    ("custom_metrics/log_red_alive_mean", "Red Alive Mean"),
    ("custom_metrics/log_blue_alive_mean", "Blue Alive Mean"),
    ("custom_metrics/log_attack_ratio_mean", "Attack Ratio"),
    ("custom_metrics/log_evade_ratio_mean", "Evade Ratio"),
    ("episode_len_mean", "Episode Length Mean"),
]

ATTACK_RATIO_KEY = "custom_metrics/log_attack_ratio_mean"
EVADE_RATIO_KEY = "custom_metrics/log_evade_ratio_mean"
LOW_LEVEL_ACTION_KEYS = [
    f"custom_metrics/log_low_action_{action_id:02d}_ratio_mean"
    for action_id in range(15)
]
LOW_LEVEL_ACTION_LABELS = [
    "0 const level",
    "1 accel level",
    "2 decel level",
    "3 const climb",
    "4 accel climb",
    "5 decel climb",
    "6 const dive",
    "7 accel dive",
    "8 decel dive",
    "9 const left",
    "10 accel left",
    "11 decel left",
    "12 const right",
    "13 accel right",
    "14 decel right",
]


def _to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_progress(progress_csv: Path) -> tuple[list[float], dict[str, list[float | None]]]:
    with progress_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {progress_csv}")

    x_key = "training_iteration" if "training_iteration" in rows[0] else None
    if x_key is None:
        x_values = list(range(1, len(rows) + 1))
    else:
        x_values = [int(float(row[x_key])) for row in rows]

    metric_keys = [key for key, _label in METRICS] + LOW_LEVEL_ACTION_KEYS
    metric_values = {}
    for key in metric_keys:
        metric_values[key] = [_to_float(row.get(key, "")) for row in rows]

    return x_values, metric_values


def plot_metrics(progress_csv: Path, output_path: Path) -> Path:
    x_values, metric_values = load_progress(progress_csv)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[idx]
        y_values = metric_values[metric_key]
        valid_points = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
        if not valid_points:
            ax.set_title(f"{metric_label} (missing)")
            ax.axis("off")
            continue

        xs = [item[0] for item in valid_points]
        ys = [item[1] for item in valid_points]
        ax.plot(xs, ys, linewidth=1.8, color="#1f77b4")
        ax.set_title(metric_label)
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)

    for idx in range(len(METRICS), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Training Metrics\n{progress_csv}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _valid_xy(x_values: list[float], y_values: list[float | None]) -> tuple[list[float], list[float]]:
    valid_points = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
    return [item[0] for item in valid_points], [item[1] for item in valid_points]


def plot_high_level_decisions(progress_csv: Path, output_path: Path, final_window: int = 10) -> Path | None:
    x_values, metric_values = load_progress(progress_csv)
    attack_x, attack_y = _valid_xy(x_values, metric_values[ATTACK_RATIO_KEY])
    evade_x, evade_y = _valid_xy(x_values, metric_values[EVADE_RATIO_KEY])

    if not attack_y or not evade_y:
        return None

    common_len = min(len(attack_x), len(evade_x))
    attack_x = attack_x[-common_len:]
    attack_y = attack_y[-common_len:]
    evade_y = evade_y[-common_len:]

    window = max(1, min(final_window, common_len))
    final_attack = sum(attack_y[-window:]) / window
    final_evade = sum(evade_y[-window:]) / window

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.2, 1.0]})

    axes[0].stackplot(
        attack_x,
        attack_y,
        evade_y,
        labels=["Attack", "Evade"],
        colors=["#d62728", "#1f77b4"],
        alpha=0.82,
    )
    axes[0].plot(attack_x, attack_y, color="#8c1d18", linewidth=1.5)
    axes[0].plot(attack_x, evade_y, color="#0d4f8b", linewidth=1.5)
    axes[0].set_title("High-Level Decision Ratio During Training")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Decision Ratio")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="upper right")

    bars = axes[1].bar(
        ["Attack", "Evade"],
        [final_attack, final_evade],
        color=["#d62728", "#1f77b4"],
        alpha=0.88,
    )
    axes[1].set_title(f"Final {window}-Iter Average")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.28)
    for bar, value in zip(bars, [final_attack, final_evade]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.suptitle(
        f"High-Level Policy Decision Distribution\nsource: {progress_csv.parent.name}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_low_level_actions(progress_csv: Path, output_path: Path, final_window: int = 10) -> Path | None:
    x_values, metric_values = load_progress(progress_csv)
    action_values = []
    for key in LOW_LEVEL_ACTION_KEYS:
        values = metric_values.get(key, [])
        if not any(value is not None for value in values):
            return None
        action_values.append([0.0 if value is None else value for value in values])

    if not action_values:
        return None

    action_array = np.asarray(action_values, dtype=float)
    if action_array.size == 0:
        return None

    window = max(1, min(final_window, action_array.shape[1]))
    final_values = action_array[:, -window:].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.25, 1.45]})

    im = axes[0].imshow(
        action_array,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Reds",
        vmin=0.0,
        vmax=max(0.01, float(action_array.max())),
    )
    axes[0].set_title("Low-Level Action Ratio Heatmap")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Discrete Action")
    tick_count = min(8, len(x_values))
    tick_indices = np.linspace(0, len(x_values) - 1, tick_count, dtype=int)
    axes[0].set_xticks(tick_indices)
    axes[0].set_xticklabels([str(x_values[idx]) for idx in tick_indices], rotation=35, ha="right")
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([str(i) for i in range(15)])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Action Ratio")

    bars = axes[1].bar(range(15), final_values, color="#d62728", alpha=0.86)
    axes[1].set_title(f"Final {window}-Iter Low-Level Action Distribution")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0.0, max(0.1, float(final_values.max()) * 1.18))
    axes[1].set_xticks(range(15))
    axes[1].set_xticklabels(LOW_LEVEL_ACTION_LABELS, rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.28)
    for bar, value in zip(bars, final_values):
        if value <= 0:
            continue
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(0.005, float(final_values.max()) * 0.02),
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"Low-Level Discrete Action Distribution\nsource: {progress_csv.parent.name}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot key RL training metrics from a progress.csv file.")
    parser.add_argument("progress_csv", type=Path, help="Path to progress.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <output_dir>/<trial_name>_metrics.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used when --output is not provided.",
    )
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=None,
        help="Optional output path for the high-level attack/evade decision ratio figure.",
    )
    parser.add_argument(
        "--low-action-output",
        type=Path,
        default=None,
        help="Optional output path for the low-level 15-action distribution figure.",
    )
    args = parser.parse_args()

    if args.output is None:
        trial_name = args.progress_csv.parent.name
        output_path = args.output_dir / f"{trial_name}_metrics.png"
    else:
        output_path = args.output

    saved_path = plot_metrics(args.progress_csv, output_path)
    print(saved_path)

    decision_output = args.decision_output
    if decision_output is None:
        trial_name = args.progress_csv.parent.name
        decision_output = args.output_dir / f"{trial_name}_high_level_decisions.png"
    decision_path = plot_high_level_decisions(args.progress_csv, decision_output)
    if decision_path is not None:
        print(decision_path)

    low_action_output = args.low_action_output
    if low_action_output is None:
        trial_name = args.progress_csv.parent.name
        low_action_output = args.output_dir / f"{trial_name}_low_level_actions.png"
    low_action_path = plot_low_level_actions(args.progress_csv, low_action_output)
    if low_action_path is not None:
        print(low_action_path)


if __name__ == "__main__":
    main()
