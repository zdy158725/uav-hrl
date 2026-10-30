#!/usr/bin/env python
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from ray.rllib.policy.policy import Policy

from exp_article_test.maneuver import (
    evade_script_action,
    pursuit_script_action,
    select_candidate_targets,
    select_coordinated_script_target,
    select_discrete_action,
    select_most_dangerous_threat,
    should_blue_evade,
)
from exp_article_test.scene import BattleScene
from exp_diffusion_hrl.config import DiffusionHRLConfig
from exp_diffusion_hrl.env import DiffusionAugmentedBattleEnv


def load_policy(checkpoint):
    path = Path(checkpoint)
    loaded = Policy.from_checkpoint(str(path), policy_ids=["red_policy"])
    return loaded["red_policy"] if isinstance(loaded, dict) else loaded


def swap_spawn_geometry(env):
    red_positions = [np.asarray(uav.pos).copy() for uav in env.scene.bf.red_uav]
    blue_positions = [np.asarray(uav.pos).copy() for uav in env.scene.bf.blue_uav]
    for index, red in enumerate(env.scene.bf.red_uav):
        red.reset(blue_positions[index % len(blue_positions)], 200, np.pi)
    for index, blue in enumerate(env.scene.bf.blue_uav):
        blue.reset(red_positions[index % len(red_positions)], 200, 0.0)
    env.scene._prediction_cache_time = {}


def prediction_errors(scene, snapshots):
    errors = {10: [], 20: [], 30: []}
    for enemy, anchor_index, prediction in snapshots:
        for step in errors:
            index = anchor_index + step
            if index >= len(enemy.record_pos):
                continue
            actual_rel = np.asarray(enemy.record_pos[index]) - np.asarray(enemy.record_pos[anchor_index])
            errors[step].append(float(np.linalg.norm(prediction[step - 1] - actual_rel)))
    return errors


def run_episode(env, policy, seed, max_steps, swapped_spawn=False):
    random.seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    if swapped_spawn:
        swap_spawn_geometry(env)
        obs = env.scene.observation()
    reward_total = 0.0
    attack = evade = 0
    all_errors = {10: [], 20: [], 30: []}
    ready_values, latency_values = [], []
    for episode_length in range(1, max_steps + 1):
        actions = {}
        for agent_id, value in obs.items():
            action = policy.compute_single_action(np.asarray(value, dtype=np.float32), explore=False)
            action = action[0] if isinstance(action, tuple) else action
            actions[agent_id] = int(action)
            attack += int(action > 0)
            evade += int(action <= 0)
        snapshots = []
        for enemy in env.scene.bf.blue_uav:
            prediction = env.scene._prediction_cache.get("Blue", {}).get(enemy.id)
            if prediction is not None and len(enemy.record_pos) >= env.scene.predictor.history_steps:
                snapshots.append((enemy, len(enemy.record_pos) - 1, prediction.copy()))
        ready_values.append(env.scene.predictor.last_ready_ratio)
        latency_values.append(env.scene.predictor.last_latency_ms)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        reward_total += sum(rewards.values())
        step_errors = prediction_errors(env.scene, snapshots)
        for step, values in step_errors.items():
            all_errors[step].extend(values)
        if terminated.get("__all__") or truncated.get("__all__"):
            break
    winner, red_alive, blue_alive = env.call_back_winner()
    decisions = attack + evade
    ade_values = all_errors[10] + all_errors[20] + all_errors[30]
    return {
        "seed": seed, "winner": winner, "red_win": int(winner == "red"),
        "blue_win": int(winner == "blue"), "draw": int(winner not in ("red", "blue")),
        "policy_win": int(winner == "red"), "opponent_win": int(winner == "blue"),
        "episode_reward": reward_total, "episode_length": episode_length,
        "red_alive": red_alive, "blue_alive": blue_alive,
        "policy_alive": red_alive, "opponent_alive": blue_alive,
        "attack_ratio": attack / decisions if decisions else 0.0,
        "evade_ratio": evade / decisions if decisions else 0.0,
        "prediction_ade": float(np.mean(ade_values)) if ade_values else np.nan,
        "prediction_fde": float(np.mean(all_errors[30])) if all_errors[30] else np.nan,
        "prediction_ready_ratio": float(np.mean(ready_values)) if ready_values else 0.0,
        "prediction_latency_ms": float(np.mean(latency_values)) if latency_values else 0.0,
    }


def improved_script_actions_for_team(battlefield, team):
    uavs = battlefield.red_uav if team == "Red" else battlefield.blue_uav
    prefix = team.lower()
    actions, assigned = {}, {}
    attack_agent_ids = set()
    attack_targets = {}
    for uav in sorted((item for item in uavs if item.alive), key=lambda item: item.blood):
        target = select_coordinated_script_target(uav, battlefield, assigned)
        threat, _ = select_most_dangerous_threat(uav, battlefield)
        if target is None and threat is None:
            actions[f"{prefix}_{uav.id}"] = 0
        elif should_blue_evade(uav, target, threat, battlefield):
            actions[f"{prefix}_{uav.id}"] = select_discrete_action(evade_script_action(uav, threat))
        elif target is None:
            actions[f"{prefix}_{uav.id}"] = 0
        else:
            agent_id = f"{prefix}_{uav.id}"
            actions[agent_id] = select_discrete_action(pursuit_script_action(uav, target))
            attack_agent_ids.add(agent_id)
            attack_targets[agent_id] = target.id
            assigned[target.id] = assigned.get(target.id, 0) + 1
    return actions, attack_agent_ids, attack_targets


def lock_macro_targets(scene, team, macro_actions):
    uavs = scene.bf.red_uav if team == "Red" else scene.bf.blue_uav
    locked_targets = {}
    for uav in uavs:
        agent_id = f"{team.lower()}_{uav.id}"
        action = int(macro_actions.get(agent_id, 0))
        if not uav.alive or action <= 0:
            continue
        candidates = select_candidate_targets(uav, scene.bf, top_k=scene.candidate_enemy_count)
        if candidates:
            locked_targets[agent_id] = candidates[min(action - 1, len(candidates) - 1)].id
    return locked_targets


def macro_to_low_actions(scene, team, macro_actions, locked_targets):
    uavs = scene.bf.red_uav if team == "Red" else scene.bf.blue_uav
    prefix = team.lower()
    result = {}
    attack_agent_ids = set()
    attack_targets = {}
    for uav in uavs:
        if not uav.alive:
            continue
        action = int(macro_actions.get(f"{prefix}_{uav.id}", 0))
        agent_id = f"{prefix}_{uav.id}"
        target_id = locked_targets.get(agent_id)
        target_team = "Blue" if team == "Red" else "Red"
        locked_target = scene._find_uav_by_id(target_team, target_id)
        if locked_target is not None and not locked_target.alive:
            locked_target = None
        low_action, target = scene.infer_low_level_action(
            mode=1 if action > 0 and locked_target is not None else 0,
            target_idx=max(0, action - 1),
            uav=uav,
            target=locked_target,
        )
        result[agent_id] = int(low_action)
        if action > 0 and target is not None:
            attack_agent_ids.add(agent_id)
            attack_targets[agent_id] = target.id
    return result, attack_agent_ids, attack_targets


def run_true_side_swap_episode(env, learned_policy, opponent_policy, opponent, seed, max_steps):
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    attack = evade = 0
    reward_total = 0.0
    ready_values, latency_values = [], []
    all_errors = {10: [], 20: [], 30: []}
    for episode_length in range(1, max_steps + 1):
        blue_macro = {}
        for blue in env.scene.bf.blue_uav:
            if not blue.alive:
                continue
            observation = env.scene.get_augmented_high_level_obs(blue)
            action = learned_policy.compute_single_action(observation, explore=False)
            action = int(action[0] if isinstance(action, tuple) else action)
            blue_macro[f"blue_{blue.id}"] = action
            attack += int(action > 0)
            evade += int(action <= 0)
        red_macro = {}
        if opponent == "pi0":
            for red in env.scene.bf.red_uav:
                if not red.alive:
                    continue
                observation = BattleScene.get_high_level_obs(env.scene, red)
                action = opponent_policy.compute_single_action(observation, explore=False)
                red_macro[f"red_{red.id}"] = int(action[0] if isinstance(action, tuple) else action)
        snapshots = []
        for red in env.scene.bf.red_uav:
            prediction = env.scene._prediction_cache.get("Red", {}).get(red.id)
            if prediction is not None and len(red.record_pos) >= env.scene.predictor.history_steps:
                snapshots.append((red, len(red.record_pos) - 1, prediction.copy()))
        ready_values.append(env.scene.predictor.last_ready_ratio)
        latency_values.append(env.scene.predictor.last_latency_ms)
        start_red, start_blue = env.scene.bf.num_red_alive, env.scene.bf.num_blue_alive
        red_locked_targets = lock_macro_targets(env.scene, "Red", red_macro)
        blue_locked_targets = lock_macro_targets(env.scene, "Blue", blue_macro)
        for _ in range(env.scene.sub_steps):
            red_actions, red_attack_agent_ids, red_attack_targets = (
                improved_script_actions_for_team(env.scene.bf, "Red")
                if opponent == "script" else macro_to_low_actions(
                    env.scene, "Red", red_macro, red_locked_targets
                )
            )
            blue_actions, blue_attack_agent_ids, blue_attack_targets = macro_to_low_actions(
                env.scene, "Blue", blue_macro, blue_locked_targets
            )
            env.scene.bf.step(
                red_actions,
                blue_actions,
                attack_agent_ids=red_attack_agent_ids | blue_attack_agent_ids,
                attack_targets={**red_attack_targets, **blue_attack_targets},
            )
            if env.scene.bf.battlefield_done["__all__"]:
                break
        reward_total += (start_red - env.scene.bf.num_red_alive) * 10.0
        reward_total -= (start_blue - env.scene.bf.num_blue_alive) * 10.0
        step_errors = prediction_errors(env.scene, snapshots)
        for step, values in step_errors.items():
            all_errors[step].extend(values)
        if env.scene.bf.battlefield_done["__all__"]:
            break
    winner, red_alive, blue_alive = env.call_back_winner()
    reward_total += 30.0 if winner == "blue" else (-30.0 if winner == "red" else 0.0)
    decisions = attack + evade
    ade_values = all_errors[10] + all_errors[20] + all_errors[30]
    return {
        "seed": seed, "winner": winner, "red_win": int(winner == "red"),
        "blue_win": int(winner == "blue"), "draw": int(winner not in ("red", "blue")),
        "policy_win": int(winner == "blue"), "opponent_win": int(winner == "red"),
        "episode_reward": reward_total, "episode_length": episode_length,
        "red_alive": red_alive, "blue_alive": blue_alive,
        "policy_alive": blue_alive, "opponent_alive": red_alive,
        "attack_ratio": attack / decisions if decisions else 0.0,
        "evade_ratio": evade / decisions if decisions else 0.0,
        "prediction_ade": float(np.mean(ade_values)) if ade_values else np.nan,
        "prediction_fde": float(np.mean(all_errors[30])) if all_errors[30] else np.nan,
        "prediction_ready_ratio": float(np.mean(ready_values)) if ready_values else 0.0,
        "prediction_latency_ms": float(np.mean(latency_values)) if latency_values else 0.0,
    }


def summarize(rows, opponent, side):
    wins = sum(row["policy_win"] for row in rows)
    losses = sum(row["opponent_win"] for row in rows)
    decisive = wins + losses
    result = {
        "opponent": opponent, "side": side, "episodes": len(rows),
        "win_rate": wins / len(rows),
        "win_rate_no_draw": wins / decisive if decisive else 0.0,
        "draw_rate": sum(row["draw"] for row in rows) / len(rows),
    }
    for key in (
        "episode_reward", "episode_length", "red_alive", "blue_alive", "policy_alive",
        "opponent_alive", "attack_ratio",
        "evade_ratio", "prediction_ade", "prediction_fde", "prediction_ready_ratio",
        "prediction_latency_ms",
    ):
        result[key] = float(np.nanmean([row[key] for row in rows]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-bundle", default=DiffusionHRLConfig["baseline_bundle"])
    parser.add_argument("--predictor-bundle", default=DiffusionHRLConfig["predictor_bundle"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--output-dir", default="outputs/diffusion_hrl/evaluation")
    args = parser.parse_args()
    policy = load_policy(args.checkpoint)
    pi0_policy = load_policy(Path(args.baseline_bundle) / "high_level_checkpoint")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []
    for opponent in ("script", "pi0"):
        config = {
            "num_red": 3, "num_blue": 3, "use_script_low_level": False,
            "sub_steps": args.sub_steps,
            "low_level_policy_dir": str(Path(args.baseline_bundle) / "policies"),
            "predictor_bundle": args.predictor_bundle, "prediction_device": "cuda",
            "blue_policy_mode": "script" if opponent == "script" else "history_policy",
            "blue_opponent_checkpoint": None if opponent == "script" else str(
                Path(args.baseline_bundle) / "high_level_checkpoint"
            ),
            "fallback_to_script": opponent == "script",
        }
        env = DiffusionAugmentedBattleEnv(config)
        for side in ("policy_red", "policy_blue"):
            rows = []
            for index in range(args.episodes):
                if side == "policy_red":
                    row = run_episode(env, policy, args.seed + index, args.max_steps)
                else:
                    row = run_true_side_swap_episode(
                        env, policy, pi0_policy, opponent, args.seed + index, args.max_steps
                    )
                row.update({"opponent": opponent, "side": side})
                rows.append(row)
            all_rows.extend(rows)
            summaries.append(summarize(rows, opponent, side))
    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(output_dir / "summary.csv")


if __name__ == "__main__":
    main()
