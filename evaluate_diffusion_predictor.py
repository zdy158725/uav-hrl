#!/usr/bin/env python
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from ray.rllib.policy.policy import Policy

from exp_article_test.env import BattleEnv
from exp_diffusion_hrl.config import DiffusionHRLConfig
from exp_diffusion_hrl.inference.predictor import DiffusionEnemyPredictor


def load_policy(path):
    loaded = Policy.from_checkpoint(str(path), policy_ids=["red_policy"])
    return loaded["red_policy"] if isinstance(loaded, dict) else loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--baseline-bundle", default=DiffusionHRLConfig["baseline_bundle"])
    parser.add_argument("--predictor-bundle", default=DiffusionHRLConfig["predictor_bundle"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/diffusion_hrl/predictor_validation")
    args = parser.parse_args()
    baseline = Path(args.baseline_bundle)
    policy = load_policy(baseline / "high_level_checkpoint")
    predictor = DiffusionEnemyPredictor(args.predictor_bundle, device=args.device)
    env = BattleEnv(
        env_config={
            "num_red": 3, "num_blue": 3, "candidate_enemy_count": 2, "sub_steps": 10,
            "use_blue_script": True, "blue_policy_mode": "script",
            "use_script_low_level": False,
            "low_level_policy_dir": str(baseline / "policies"), "freeze_low_level": True,
        }
    )
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        random.seed(seed)
        np.random.seed(seed)
        observations, _ = env.reset(seed=seed)
        for macro_step in range(args.max_steps):
            actions = {}
            for agent_id, observation in observations.items():
                action = policy.compute_single_action(observation, explore=False)
                actions[agent_id] = int(action[0] if isinstance(action, tuple) else action)
            snapshots = []
            predictions = predictor.predict(env.scene.bf)
            for blue in env.scene.bf.blue_uav:
                if not blue.alive or len(blue.record_pos) < predictor.history_steps:
                    continue
                snapshots.append(
                    (
                        blue,
                        len(blue.record_pos) - 1,
                        predictions[blue.id].copy(),
                        predictor._constant_velocity(blue),
                    )
                )
            observations, _, terminated, truncated, _ = env.step(actions)
            for blue, anchor, prediction, cv_prediction in snapshots:
                for horizon in (10, 20, 30):
                    if anchor + horizon >= len(blue.record_pos):
                        continue
                    actual = np.asarray(blue.record_pos[anchor + horizon]) - np.asarray(
                        blue.record_pos[anchor]
                    )
                    rows.append(
                        {
                            "episode": episode,
                            "macro_step": macro_step,
                            "enemy_id": blue.id,
                            "horizon_s": horizon / 10.0,
                            "diffusion_error_m": np.linalg.norm(prediction[horizon - 1] - actual),
                            "constant_velocity_error_m": np.linalg.norm(
                                cv_prediction[horizon - 1] - actual
                            ),
                            "latency_ms": predictor.last_latency_ms,
                        }
                    )
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prediction_errors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for horizon in (1.0, 2.0, 3.0):
        selected = [row for row in rows if row["horizon_s"] == horizon]
        summary.append(
            {
                "horizon_s": horizon,
                "samples": len(selected),
                "diffusion_error_m": np.mean([row["diffusion_error_m"] for row in selected]),
                "constant_velocity_error_m": np.mean(
                    [row["constant_velocity_error_m"] for row in selected]
                ),
                "mean_latency_ms": np.mean([row["latency_ms"] for row in selected]),
            }
        )
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(output_dir / "summary.csv")


if __name__ == "__main__":
    main()
