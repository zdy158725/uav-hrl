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


METRICS = [
    ("episode_reward_mean", "Episode Reward Mean"),
    ("custom_metrics/log_red_win_mean_mean", "Red Win Rate"),
    ("custom_metrics/log_red_alive_mean", "Red Alive Mean"),
    ("custom_metrics/log_blue_alive_mean", "Blue Alive Mean"),
    ("custom_metrics/log_attack_ratio_mean", "Attack Ratio"),
    ("custom_metrics/log_evade_ratio_mean", "Evade Ratio"),
    ("episode_len_mean", "Episode Length Mean"),
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

    metric_values = {}
    for key, _label in METRICS:
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

    if len(axes) > len(METRICS):
        axes[-1].axis("off")

    fig.suptitle(f"Training Metrics\n{progress_csv}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

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
    args = parser.parse_args()

    if args.output is None:
        trial_name = args.progress_csv.parent.name
        output_path = args.output_dir / f"{trial_name}_metrics.png"
    else:
        output_path = args.output

    saved_path = plot_metrics(args.progress_csv, output_path)
    print(saved_path)


if __name__ == "__main__":
    main()
