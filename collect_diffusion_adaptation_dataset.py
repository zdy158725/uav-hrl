#!/usr/bin/env python
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from ray.rllib.policy.policy import Policy

from exp_article_test.env import BattleEnv
from exp_diffusion_hrl.config import DiffusionHRLConfig
from exp_diffusion_hrl.env import DiffusionAugmentedBattleEnv


CSV_COLUMNS = [
    "episode_id",
    "timestep",
    "drone_id",
    "team",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "hp",
    "time",
    "roll",
    "pitch",
    "yaw",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "airspeed",
    "speed_xyz",
    "terminal_status",
    "platform",
    "slot",
    "source_run",
    "source_episode",
    "source_file",
]


def remap_path(path):
    path = Path(str(path))
    if path.exists():
        return path
    text = str(path)
    pairs = [
        ("/home/dfc/Documents/zdy_documents", "/mnt/disk2/zdy_documents"),
        ("/mnt/disk2/zdy_documents", "/home/dfc/Documents/zdy_documents"),
    ]
    for old, new in pairs:
        if text.startswith(old):
            candidate = Path(new + text[len(old):])
            if candidate.exists():
                return candidate
    return path


def load_policy(path):
    loaded = Policy.from_checkpoint(str(path), policy_ids=["red_policy"])
    return loaded["red_policy"] if isinstance(loaded, dict) else loaded


def policy_obs_dim(policy):
    space = getattr(policy, "observation_space", None)
    shape = getattr(space, "shape", None)
    if shape:
        return int(shape[0])
    weights = policy.get_weights()
    for key, value in weights.items():
        if key.endswith("._model.0.weight") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[1])
    raise RuntimeError("Cannot infer policy observation dimension")


def make_env(obs_dim, baseline_bundle, pi0_checkpoint, predictor_bundle, prediction_device, blue_policy_mode):
    env_config = {
        "num_red": 3,
        "num_blue": 3,
        "candidate_enemy_count": 2,
        "sub_steps": 10,
        "use_blue_script": True,
        "blue_policy_mode": blue_policy_mode,
        "blue_opponent_checkpoint": None if blue_policy_mode == "script" else str(pi0_checkpoint),
        "fallback_to_script": False if blue_policy_mode == "history_policy" else True,
        "use_script_low_level": False,
        "low_level_policy_dir": str(baseline_bundle / "policies"),
        "freeze_low_level": True,
    }
    if obs_dim == 58:
        env_config.update(
            {
                "predictor_bundle": str(predictor_bundle),
                "prediction_device": prediction_device,
                "prediction_steps": (10, 20, 30),
                "num_inference_steps": 10,
            }
        )
        return DiffusionAugmentedBattleEnv(env_config)
    if obs_dim == 38:
        return BattleEnv(env_config=env_config)
    raise ValueError(f"Unsupported policy obs dim: {obs_dim}")


def compute_velocity(airspeed, theta, psi):
    ctheta = math.cos(float(theta))
    return np.asarray(
        [
            airspeed * ctheta * math.cos(float(psi)),
            airspeed * ctheta * math.sin(float(psi)),
            -airspeed * math.sin(float(theta)),
        ],
        dtype=np.float32,
    )


def angular_diff(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def uav_rows(uav, episode_id, source_run, source_episode):
    rows = []
    length = len(uav.record_pos)
    if length <= 0:
        return rows

    team = str(uav.team).upper()
    drone_id = f"{uav.team.lower()}_{uav.id}"
    slot = int(uav.id)
    terminal_status = str(uav.info)
    platform = "f16"
    source_file = f"{source_run}/{source_episode}/{drone_id}_{terminal_status}.csv"

    prev_phi = None
    prev_theta = None
    prev_psi = None
    prev_t = None
    for index in range(length):
        pos = np.asarray(uav.record_pos[index], dtype=np.float32)
        airspeed = float(uav.record_airspeed[index])
        theta = float(uav.record_theta[index])
        phi = float(uav.record_phi[index])
        psi = float(uav.record_psi[index])
        t = float(uav.record_t[index])
        blood = float(uav.record_blood[index])
        vel = compute_velocity(airspeed, theta, psi)
        dt = None if prev_t is None else max(t - prev_t, 1e-6)
        if dt is None:
            roll_rate = pitch_rate = yaw_rate = 0.0
        else:
            roll_rate = angular_diff(phi, prev_phi) / dt
            pitch_rate = angular_diff(theta, prev_theta) / dt
            yaw_rate = angular_diff(psi, prev_psi) / dt
        rows.append(
            {
                "episode_id": episode_id,
                "timestep": index,
                "drone_id": drone_id,
                "team": team,
                "x": f"{float(pos[0]):.9f}",
                "y": f"{float(pos[1]):.9f}",
                "z": f"{float(pos[2]):.9f}",
                "vx": f"{float(vel[0]):.9f}",
                "vy": f"{float(vel[1]):.9f}",
                "vz": f"{float(vel[2]):.9f}",
                "hp": f"{blood:.9f}",
                "time": f"{t:.9f}",
                "roll": f"{phi:.9f}",
                "pitch": f"{theta:.9f}",
                "yaw": f"{psi:.9f}",
                "roll_rate": f"{roll_rate:.9f}",
                "pitch_rate": f"{pitch_rate:.9f}",
                "yaw_rate": f"{yaw_rate:.9f}",
                "airspeed": f"{airspeed:.9f}",
                "speed_xyz": f"{float(np.linalg.norm(vel)):.9f}",
                "terminal_status": terminal_status,
                "platform": platform,
                "slot": slot,
                "source_run": source_run,
                "source_episode": source_episode,
                "source_file": source_file,
            }
        )
        prev_phi = phi
        prev_theta = theta
        prev_psi = psi
        prev_t = t
    return rows


def rollout_episode(env, policy, seed, max_steps):
    random.seed(seed)
    np.random.seed(seed)
    observations, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        if not observations:
            break
        actions = {}
        for agent_id, observation in observations.items():
            action = policy.compute_single_action(observation, explore=False)
            actions[agent_id] = int(action[0] if isinstance(action, tuple) else action)
        observations, _, terminated, truncated, _ = env.step(actions)
        if terminated.get("__all__") or truncated.get("__all__"):
            break
    winner, red_alive, blue_alive = env.call_back_winner()
    return {
        "winner": winner,
        "red_alive": red_alive,
        "blue_alive": blue_alive,
        "red_uav": list(env.scene.bf.red_uav),
        "blue_uav": list(env.scene.bf.blue_uav),
        "sim_time": float(env.scene.bf.t),
    }


def infer_blue_mode(tag):
    if tag.startswith("script_strong"):
        return "script"
    return "history_policy"


def write_split_stats(output_dir, episode_ids, train_ratio, seed):
    rng = np.random.RandomState(seed)
    shuffled = list(episode_ids)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        train_ids, val_ids = shuffled, []
    else:
        train_count = int(round(len(shuffled) * train_ratio))
        train_count = max(1, min(len(shuffled) - 1, train_count))
        train_ids, val_ids = shuffled[:train_count], shuffled[train_count:]
    stats = {
        "episodes_total": len(episode_ids),
        "train_scene_count": len(train_ids),
        "val_scene_count": len(val_ids),
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "dt": 0.1,
        "hp_scale": 1.0,
        "standardization_min_std": 0.001,
    }
    split_path = output_dir / "uav_3d_preprocess_stats.json"
    split_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return split_path


def read_existing_episode_ids(output_csv):
    if not output_csv.exists():
        return [], 0
    episode_ids = []
    seen = set()
    row_count = 0
    with output_csv.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_count += 1
            episode_id = row.get("episode_id")
            if episode_id and episode_id not in seen:
                seen.add(episode_id)
                episode_ids.append(episode_id)
    return episode_ids, row_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-csv",
        default="outputs/diffusion_hrl/adaptation_checkpoint_candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diffusion_hrl/adaptation_dataset",
    )
    parser.add_argument("--episodes-per-checkpoint", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--prediction-device", default="cuda")
    parser.add_argument("--baseline-bundle", default=DiffusionHRLConfig["baseline_bundle"])
    parser.add_argument("--predictor-bundle", default=DiffusionHRLConfig["predictor_bundle"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing episodes to an existing output CSV instead of overwriting it.",
    )
    args = parser.parse_args()

    candidate_csv = remap_path(args.candidate_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bundle = remap_path(args.baseline_bundle)
    predictor_bundle = remap_path(args.predictor_bundle)
    pi0_checkpoint = baseline_bundle / "high_level_checkpoint"

    with candidate_csv.open("r", newline="") as file:
        candidates = list(csv.DictReader(file))

    output_csv = output_dir / "aircombat_uav_3d_hrl_adaptation.csv"
    manifest = {
        "candidate_csv": str(candidate_csv),
        "output_csv": str(output_csv),
        "episodes_per_checkpoint": int(args.episodes_per_checkpoint),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "baseline_bundle": str(baseline_bundle),
        "pi0_checkpoint": str(pi0_checkpoint),
        "predictor_bundle": str(predictor_bundle),
        "checkpoints": [],
    }

    existing_episode_ids, existing_rows = (
        read_existing_episode_ids(output_csv) if args.resume else ([], 0)
    )
    completed_episode_ids = set(existing_episode_ids)
    if args.resume and output_csv.exists():
        print(
            f"resume enabled: found {len(existing_episode_ids)} existing episodes "
            f"and {existing_rows} existing rows"
        )

    total_rows = existing_rows
    episode_ids = list(existing_episode_ids)
    summary_rows = []
    open_mode = "a" if args.resume and output_csv.exists() else "w"
    with output_csv.open(open_mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if open_mode == "w":
            writer.writeheader()
        for ckpt_index, candidate in enumerate(candidates):
            tag = candidate["tag"]
            checkpoint_path = remap_path(candidate["checkpoint_path"])
            blue_mode = infer_blue_mode(tag)
            print(f"[{ckpt_index + 1}/{len(candidates)}] load {tag}: {checkpoint_path}")
            policy = load_policy(checkpoint_path)
            obs_dim = policy_obs_dim(policy)
            env = make_env(
                obs_dim=obs_dim,
                baseline_bundle=baseline_bundle,
                pi0_checkpoint=pi0_checkpoint,
                predictor_bundle=predictor_bundle,
                prediction_device=args.prediction_device,
                blue_policy_mode=blue_mode,
            )
            checkpoint_summary = {
                "tag": tag,
                "role": candidate.get("role", ""),
                "checkpoint_path": str(checkpoint_path),
                "obs_dim": obs_dim,
                "blue_policy_mode": blue_mode,
                "episodes": [],
            }
            for episode_index in range(int(args.episodes_per_checkpoint)):
                seed = int(args.seed) + ckpt_index * 100000 + episode_index
                source_episode = f"ep{episode_index:04d}"
                episode_id = f"{tag}__{source_episode}"
                if episode_id in completed_episode_ids:
                    continue
                result = rollout_episode(env, policy, seed=seed, max_steps=args.max_steps)
                rows = []
                for uav in result["blue_uav"] + result["red_uav"]:
                    rows.extend(
                        uav_rows(
                            uav=uav,
                            episode_id=episode_id,
                            source_run=tag,
                            source_episode=source_episode,
                        )
                    )
                writer.writerows(rows)
                total_rows += len(rows)
                episode_ids.append(episode_id)
                ep_summary = {
                    "episode_id": episode_id,
                    "seed": seed,
                    "winner": result["winner"],
                    "red_alive": int(result["red_alive"]),
                    "blue_alive": int(result["blue_alive"]),
                    "sim_time": float(result["sim_time"]),
                    "rows": len(rows),
                }
                checkpoint_summary["episodes"].append(ep_summary)
                summary_rows.append({"tag": tag, **ep_summary})
                if (episode_index + 1) % 10 == 0 or episode_index == 0:
                    print(
                        f"  {tag} episode {episode_index + 1}/{args.episodes_per_checkpoint}: "
                        f"winner={result['winner']} rows={len(rows)} total_rows={total_rows}"
                    )
            manifest["checkpoints"].append(checkpoint_summary)

    split_path = write_split_stats(output_dir, episode_ids, args.train_ratio, args.seed)
    manifest["rows_total"] = total_rows
    manifest["episodes_total"] = len(episode_ids)
    manifest["split_stats_path"] = str(split_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "episode_summary.csv").open("w", newline="") as file:
        fieldnames = [
            "tag",
            "episode_id",
            "seed",
            "winner",
            "red_alive",
            "blue_alive",
            "sim_time",
            "rows",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"dataset_csv={output_csv}")
    print(f"split_stats={split_path}")
    print(f"episodes={len(episode_ids)} rows={total_rows}")


if __name__ == "__main__":
    main()
