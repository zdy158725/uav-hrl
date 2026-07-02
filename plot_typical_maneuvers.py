from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
sys.path.insert(0, str(BASE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MarkerPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import proj3d

from exp_article_test.maneuver import convert_to_contol_input
from exp_article_test.uav import UAV


plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


AIRPLANE_MARKER = MarkerPath(
    np.array(
        [
            [0.0, 1.0],
            [0.18, 0.15],
            [0.75, -0.15],
            [0.15, -0.25],
            [0.08, -0.8],
            [0.0, -0.55],
            [-0.08, -0.8],
            [-0.15, -0.25],
            [-0.75, -0.15],
            [-0.18, 0.15],
            [0.0, 1.0],
        ],
        dtype=float,
    )
)


def _rotated_airplane_marker(angle_rad: float) -> MarkerPath:
    # The base marker nose points upward in screen coordinates.
    return Affine2D().rotate(angle_rad - np.pi / 2.0).transform_path(AIRPLANE_MARKER)


def _screen_heading(ax, x0, y0, h0, x1, y1, h1) -> float:
    p0 = proj3d.proj_transform(x0, y0, h0, ax.get_proj())
    p1 = proj3d.proj_transform(x1, y1, h1, ax.get_proj())
    return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


SPEED_VARIANTS = [
    ("accelerate", "加速", 4.0),
    ("constant", "匀速", 0.0),
    ("decelerate", "减速", -4.0),
]


TYPICAL_MANEUVER_GROUPS = [
    {
        "name": "level_flight",
        "title": "平飞",
        "actions": {"accelerate": 1, "constant": 0, "decelerate": 2},
        "steps": 65,
    },
    {
        "name": "climb",
        "title": "爬升",
        "actions": {"accelerate": 4, "constant": 3, "decelerate": 5},
        "steps": 55,
    },
    {
        "name": "dive",
        "title": "俯冲",
        "actions": {"accelerate": 7, "constant": 6, "decelerate": 8},
        "steps": 55,
    },
    {
        "name": "left_turn",
        "title": "左转",
        "actions": {"accelerate": 10, "constant": 9, "decelerate": 11},
        "steps": 60,
    },
    {
        "name": "right_turn",
        "title": "右转",
        "actions": {"accelerate": 13, "constant": 12, "decelerate": 14},
        "steps": 60,
    },
]


def simulate_action(action_id: int, steps: int, init_pos: np.ndarray, airspeed: float) -> dict[str, np.ndarray]:
    uav = UAV("Red", str(action_id))
    uav.reset(pos=init_pos.copy(), airspeed=airspeed, psi=0.0, theta=0.0, phi=0.0)
    action = convert_to_contol_input(action_id)

    for _ in range(steps):
        uav.step(action)
        uav.record_status()

    return {
        "pos": np.asarray(uav.record_pos, dtype=float),
        "airspeed": np.asarray(uav.record_airspeed, dtype=float),
        "time": np.asarray(uav.record_t, dtype=float),
    }


def set_equal_3d_axes(ax, xs, ys, hs):
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


def _set_2d_limits(ax, x_values, y_values):
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)
    ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.12 * y_span)


def _relative_trajectory(sim: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = sim["pos"]
    x = pos[:, 0] - pos[0, 0]
    y = pos[:, 1] - pos[0, 1]
    h = (-pos[:, 2]) - (-pos[0, 2])
    return x, y, h


def _sample_indices(length: int, sample_count: int) -> np.ndarray:
    sample_count = min(sample_count, length)
    return np.unique(np.linspace(0, length - 1, sample_count, dtype=int))


def plot_maneuver_group(
    group: dict,
    output_path: Path,
    steps: int | None = None,
    airspeed: float = 200.0,
    sample_count: int = 6,
) -> Path:
    init_pos = np.array([8000.0, 8000.0, -4500.0], dtype=float)
    group_steps = steps if steps is not None else group["steps"]
    sims = {
        variant_name: simulate_action(
            group["actions"][variant_name],
            steps=group_steps,
            init_pos=init_pos,
            airspeed=airspeed,
        )
        for variant_name, _variant_label, _nx in SPEED_VARIANTS
    }

    rel = {variant_name: _relative_trajectory(sim) for variant_name, sim in sims.items()}
    all_x = np.concatenate([values[0] for values in rel.values()])
    all_y = np.concatenate([values[1] for values in rel.values()])
    all_h = np.concatenate([values[2] for values in rel.values()])
    fig = plt.figure(figsize=(18.5, 6.4))
    fig.suptitle(f"{group['title']}：加速 / 匀速 / 减速", y=0.96, fontsize=15)
    plot_axes = []

    for idx, (variant_name, variant_label, nx_value) in enumerate(SPEED_VARIANTS, start=1):
        action_id = group["actions"][variant_name]
        sim = sims[variant_name]
        x, y, h = rel[variant_name]
        speeds = sim["airspeed"]
        sample_idx = _sample_indices(len(x), sample_count)
        sample_idx = sample_idx[(sample_idx != 0) & (sample_idx != len(x) - 1)]

        ax = fig.add_subplot(1, 3, idx, projection="3d")
        plot_axes.append(ax)
        ax.plot(x, y, h, linewidth=2.4, color="#d62728", alpha=0.95)
        ax.set_title(
            f"{variant_label} | 动作 {action_id} | nx={nx_value:g}\n"
            f"速度：{speeds[0]:.0f}→{speeds[-1]:.0f} m/s"
        )
        ax.set_xlabel("纵向位移 Δx / m")
        ax.set_ylabel("侧向位移 Δy / m")
        ax.set_zlabel("高度变化 Δh / m")
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, alpha=0.3)
        set_equal_3d_axes(ax, all_x, all_y, all_h)

        ax.scatter(
            x[sample_idx],
            y[sample_idx],
            h[sample_idx],
            color="#d62728",
            s=24,
            depthshade=True,
        )
        start_heading = _screen_heading(ax, x[0], y[0], h[0], x[1], y[1], h[1])
        end_heading = _screen_heading(ax, x[-2], y[-2], h[-2], x[-1], y[-1], h[-1])
        ax.scatter(
            x[0],
            y[0],
            h[0],
            marker=_rotated_airplane_marker(start_heading),
            s=230,
            color="#d62728",
            edgecolor="#111111",
            linewidth=0.6,
            depthshade=True,
        )
        ax.scatter(
            x[-1],
            y[-1],
            h[-1],
            marker=_rotated_airplane_marker(end_heading),
            s=230,
            color="#d62728",
            edgecolor="#111111",
            linewidth=0.6,
            depthshade=True,
        )

    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.04, top=0.78, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_typical_maneuvers(
    output_dir: Path,
    steps: int | None = None,
    airspeed: float = 200.0,
    sample_count: int = 14,
) -> list[Path]:
    saved_paths = []
    for group in TYPICAL_MANEUVER_GROUPS:
        output_path = output_dir / f"{group['name']}_speed_variants_3d.png"
        saved_paths.append(
            plot_maneuver_group(
                group=group,
                output_path=output_path,
                steps=steps,
                airspeed=airspeed,
                sample_count=sample_count,
            )
        )
    return saved_paths


def plot_combined_maneuvers(output_path: Path, steps: int = 120, airspeed: float = 200.0) -> Path:
    init_pos = np.array([8000.0, 8000.0, -4500.0], dtype=float)
    trajectories = {
        group["name"]: simulate_action(
            group["actions"]["constant"],
            steps=steps,
            init_pos=init_pos,
            airspeed=airspeed,
        )
        for group in TYPICAL_MANEUVER_GROUPS
    }

    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    all_x, all_y, all_h = [], [], []
    for group in TYPICAL_MANEUVER_GROUPS:
        name = group["name"]
        action_id = group["actions"]["constant"]
        pos = trajectories[name]["pos"]
        x = pos[:, 0]
        y = pos[:, 1]
        h = -pos[:, 2]
        all_x.append(x)
        all_y.append(y)
        all_h.append(h)

        label = f"{group['title']}（动作 {action_id}）"
        ax3d.plot(x, y, h, linewidth=2.0, label=label)
        ax3d.scatter(x[0], y[0], h[0], marker="o", s=18)
        ax3d.scatter(x[-1], y[-1], h[-1], marker="x", s=35)

        ax2d.plot(x, y, linewidth=2.0, label=label)
        ax2d.scatter(x[0], y[0], marker="o", s=18)
        ax2d.scatter(x[-1], y[-1], marker="x", s=35)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_h = np.concatenate(all_h)

    ax3d.set_title("典型低层机动轨迹")
    ax3d.set_xlabel("x / m")
    ax3d.set_ylabel("y / m")
    ax3d.set_zlabel("高度 h=-z / m")
    ax3d.view_init(elev=24, azim=-55)
    ax3d.legend(fontsize=8)
    set_equal_3d_axes(ax3d, all_x, all_y, all_h)

    ax2d.set_title("俯视 x-y 投影")
    ax2d.set_xlabel("x / m")
    ax2d.set_ylabel("y / m")
    ax2d.grid(True, alpha=0.3)
    ax2d.axis("equal")
    ax2d.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot trajectories for typical low-level maneuver actions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs" / "figures" / "typical_maneuvers",
        help="Output directory for individual maneuver images.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override simulation steps at dt=0.1s.")
    parser.add_argument("--airspeed", type=float, default=200.0, help="Initial airspeed in m/s.")
    parser.add_argument("--sample-count", type=int, default=6, help="Number of marked samples per trajectory.")
    parser.add_argument("--combined", action="store_true", help="Also save a combined overview figure.")
    args = parser.parse_args()

    saved_paths = plot_typical_maneuvers(
        args.output_dir,
        steps=args.steps,
        airspeed=args.airspeed,
        sample_count=args.sample_count,
    )
    if args.combined:
        saved_paths.append(
            plot_combined_maneuvers(args.output_dir / "combined_overview.png", airspeed=args.airspeed)
        )
    for saved_path in saved_paths:
        print(saved_path)


if __name__ == "__main__":
    main()
