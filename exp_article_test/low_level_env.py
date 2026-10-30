from pathlib import Path

import numpy as np
import torch
from ray.rllib.env.multi_agent_env import MultiAgentEnv

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

from exp_article_test.battlefield import BattleField
from exp_article_test.config import LowLevelConfig
from exp_article_test.maneuver import (
    ACTION_COUNT,
    make_decision_for_blue_team,
    option_attack,
    option_evade,
    select_candidate_targets,
)
from exp_article_test.toolkit import get_rela_angle


def _self_features(uav):
    psi_norm = (np.arctan2(np.sin(uav.psi), np.cos(uav.psi)) + np.pi) / (2 * np.pi)
    return np.array(
        [
            uav.pos[0] / 16000.0,
            uav.pos[1] / 16000.0,
            -uav.pos[2] / 8000.0,
            uav.vel[0] / 250.0,
            uav.vel[1] / 250.0,
            uav.vel[2] / 250.0,
            uav.blood,
            psi_norm,
        ],
        dtype=np.float32,
    )


def _enemy_features(uav, enemy):
    if enemy is None or not enemy.alive:
        return np.zeros(11, dtype=np.float32)

    rel_pos = enemy.pos - uav.pos
    rel_vel = enemy.vel - uav.vel
    my_dir = uav.vel / (np.linalg.norm(uav.vel) + 1e-6)
    enemy_dir = enemy.vel / (np.linalg.norm(enemy.vel) + 1e-6)
    rel_dir = rel_pos / (np.linalg.norm(rel_pos) + 1e-6)
    my_angle = np.arccos(np.clip(np.dot(rel_dir, my_dir), -1.0, 1.0))
    enemy_angle = np.arccos(np.clip(np.dot(-rel_dir, enemy_dir), -1.0, 1.0))
    dist = np.linalg.norm(rel_pos) / 16000.0
    angle_adv = 1.0 - (my_angle / np.pi)
    return np.array(
        [
            rel_pos[0] / 16000.0,
            rel_pos[1] / 16000.0,
            rel_pos[2] / 8000.0,
            rel_vel[0] / 250.0,
            rel_vel[1] / 250.0,
            rel_vel[2] / 250.0,
            dist,
            my_angle / np.pi,
            enemy_angle / np.pi,
            angle_adv,
            enemy.blood,
        ],
        dtype=np.float32,
    )


def _ally_features(uav, bf):
    allies = []
    for ally in bf.red_uav if uav.team == "Red" else bf.blue_uav:
        if ally.id == uav.id or not ally.alive:
            continue
        allies.append((ally, np.linalg.norm(ally.pos - uav.pos)))
    if not allies:
        return np.zeros(8, dtype=np.float32)
    ally = min(allies, key=lambda item: item[1])[0]
    rel_pos = ally.pos - uav.pos
    rel_vel = ally.vel - uav.vel
    dist = np.linalg.norm(rel_pos) / 16000.0
    return np.array(
        [
            rel_pos[0] / 16000.0,
            rel_pos[1] / 16000.0,
            rel_pos[2] / 8000.0,
            rel_vel[0] / 250.0,
            rel_vel[1] / 250.0,
            rel_vel[2] / 250.0,
            dist,
            ally.blood,
        ],
        dtype=np.float32,
    )


def get_uav_advantage(uav, enemy):
    if enemy is None or not enemy.alive:
        return 0.0
    rel = enemy.pos - uav.pos
    dist = np.linalg.norm(rel)
    my_angle = get_rela_angle(uav.vel, rel)
    enemy_angle = get_rela_angle(enemy.vel, -rel)
    dist_score = np.exp(-max(0.0, dist - 1000.0) / 3000.0)
    angle_score = 1.0 - (my_angle / np.pi)
    pressure = 1.0 - (enemy_angle / np.pi)
    return angle_score * dist_score - pressure


def build_low_level_obs(mode, uav, bf, target=None, threat_count=None):
    threat_count = threat_count or LowLevelConfig["threat_count"]
    obs_parts = [_self_features(uav)]
    if mode == "attack":
        obs_parts.append(_enemy_features(uav, target))
        obs_parts.append(_ally_features(uav, bf))
    else:
        threats = select_candidate_targets(uav, bf, top_k=threat_count)
        for idx in range(threat_count):
            obs_parts.append(_enemy_features(uav, threats[idx] if idx < len(threats) else None))
        obs_parts.append(_ally_features(uav, bf))
    return np.concatenate(obs_parts).astype(np.float32)


class LowLevelController:
    def __init__(self, use_script=True, policy_dir=None, device="cpu"):
        self.use_script = use_script
        self.policy_dir = Path(policy_dir) if policy_dir else None
        self.device = torch.device(device)
        self.models = {}
        if not self.use_script and self.policy_dir:
            self._load_models()

    def _load_models(self):
        attack_path = self.policy_dir / "low_attack_policy.pt"
        evade_path = self.policy_dir / "low_evade_policy.pt"
        if attack_path.exists():
            self.models["attack"] = self._load_model(attack_path)
        if evade_path.exists():
            self.models["evade"] = self._load_model(evade_path)

    def _load_model(self, path):
        model = torch.load(path, map_location=self.device)
        model.eval()
        return model

    def _policy_action(self, mode, obs):
        model = self.models.get(mode)
        if model is None:
            raise FileNotFoundError(f"Missing low-level policy for mode={mode}")
        with torch.no_grad():
            logits, _ = model(
                {"obs": torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)},
                [],
                None,
            )
        if logits.shape[-1] != ACTION_COUNT:
            raise ValueError(
                f"Low-level {mode} policy outputs {logits.shape[-1]} actions, "
                f"but the current action library has {ACTION_COUNT}; use a 15-action policy."
            )
        return int(torch.argmax(logits[0]).item())

    def infer_action(self, mode, uav, bf, target=None):
        if self.use_script or mode not in self.models:
            if mode == "attack":
                action = option_attack(uav, target, bf)
            else:
                threats = select_candidate_targets(uav, bf, top_k=LowLevelConfig["threat_count"])
                action = option_evade(uav, threats, bf)
            return int(0 if action is None else action)

        obs = build_low_level_obs(mode, uav, bf, target=target, threat_count=LowLevelConfig["threat_count"])
        return self._policy_action(mode, obs)


class LowLevelBattleEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        env_config = env_config or {}
        self.mode = env_config.get("mode", "attack")
        self.num_red = env_config.get("num_red", LowLevelConfig["start_train_num_red"])
        self.num_blue = env_config.get("num_blue", LowLevelConfig["start_train_num_blue"])
        self.use_blue_script = env_config.get("use_blue_script", True)
        self.blue_policy_mode = env_config.get("blue_policy_mode", "script")
        self.threat_count = env_config.get("threat_count", LowLevelConfig["threat_count"])

        self.possible_agents = [f"red_{i}" for i in range(self.num_red)]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.possible_agents)
        self.blue_scene = None
        if self.blue_policy_mode == "history_policy":
            from exp_article_test.scene import BattleScene

            self.blue_scene = BattleScene(
                self.num_red, self.num_blue,
                use_script_low_level=True,
                low_level_policy_dir=env_config.get("blue_low_level_policy_dir"),
                blue_low_level_policy_dir=env_config.get("blue_low_level_policy_dir"),
                blue_high_level_sub_steps=env_config.get("blue_high_level_sub_steps", 1),
                blue_policy_mode="history_policy",
                blue_opponent_checkpoint=env_config.get("blue_opponent_checkpoint"),
                fallback_to_script=False,
            )
            self.bf = self.blue_scene.bf
        else:
            self.bf = BattleField(self.num_red, self.num_blue)
        self.assigned_targets = {}
        self.prev_status = {}
        self.prev_distances = {}
        self.prev_red_alive = self.num_red
        self.prev_blue_alive = self.num_blue
        self.last_rewards = {}
        self.episode_stats = {
            "mode": self.mode,
            "kill_events": 0,
            "death_events": 0,
            "out_events": 0,
        }

        obs_dim = LowLevelConfig["attack_obs_dim"] if self.mode == "attack" else LowLevelConfig["evade_obs_dim"]
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_spaces = {agent: self.observation_space for agent in self.agents}
        self.action_spaces = {agent: self.action_space for agent in self.agents}
        super().__init__()

    def reset(self, *, seed=None, options=None):
        del seed, options
        if self.blue_scene is not None:
            self.blue_scene.reset()
        else:
            self.bf.reset()
        self.assigned_targets = {}
        self.prev_distances = {}
        self.prev_red_alive = self.num_red
        self.prev_blue_alive = self.num_blue
        self.last_rewards = {}
        self.episode_stats = {
            "mode": self.mode,
            "kill_events": 0,
            "death_events": 0,
            "out_events": 0,
            "low_action_counts": [0] * ACTION_COUNT,
        }
        self._assign_targets()
        self.prev_status = self._snapshot_status()
        self._snapshot_distances()
        obs = self.observation()
        self.agents = list(obs.keys())
        info = self.info()
        return obs, info

    def step(self, action_dict):
        self.prev_status = self._snapshot_status()
        self.prev_red_alive = self.bf.num_red_alive
        self.prev_blue_alive = self.bf.num_blue_alive
        self._snapshot_distances()
        if self.blue_scene is not None:
            blue_actions, blue_attack_agent_ids, blue_attack_targets = self.blue_scene._make_blue_actions()
        elif self.use_blue_script:
            blue_actions, blue_attack_agent_ids, blue_attack_targets = make_decision_for_blue_team(
                self.bf, return_targets=True,
            )
        else:
            blue_actions, blue_attack_agent_ids, blue_attack_targets = {}, set(), {}
        red_attack_targets = {}
        if self.mode == "attack":
            for red in self.bf.red_uav:
                agent_id = f"red_{red.id}"
                if red.alive and agent_id in action_dict:
                    target = self._current_target(red)
                    if target is not None:
                        red_attack_targets[agent_id] = target.id
        red_attack_agent_ids = set(red_attack_targets)
        self._record_low_level_actions(action_dict)
        self.bf.step(
            action_dict,
            blue_actions,
            attack_agent_ids=red_attack_agent_ids | blue_attack_agent_ids,
            attack_targets={**red_attack_targets, **blue_attack_targets},
        )
        self._assign_targets()
        self.last_rewards = self._build_rewards()
        obs = self.observation()
        rewards = self.reward()
        terminated = self.done()
        truncated = self.truncated()
        info = self.info()
        self.agents = list(obs.keys())
        return obs, rewards, terminated, truncated, info

    def observation(self):
        obs = {}
        for red in self.bf.red_uav:
            agent_id = f"red_{red.id}"
            if not self.bf.battlefield_done[agent_id]:
                target = self._current_target(red)
                obs[agent_id] = build_low_level_obs(self.mode, red, self.bf, target=target, threat_count=self.threat_count)
        return obs

    def reward(self):
        return {agent_id: float(value) for agent_id, value in self.last_rewards.items()}

    def done(self):
        return self.bf.battlefield_done

    def truncated(self):
        truncated = {agent: False for agent in self.possible_agents}
        truncated["__all__"] = False
        return truncated

    def info(self):
        info = {}
        for red in self.bf.red_uav:
            agent_id = f"red_{red.id}"
            if not self.bf.battlefield_done[agent_id]:
                info[agent_id] = {
                    "mode": self.mode,
                    "target_id": self.assigned_targets.get(agent_id),
                }
        return info

    def _snapshot_status(self):
        return {
            f"red_{uav.id}": {"alive": uav.alive, "info": uav.info}
            for uav in self.bf.red_uav
        }

    def _snapshot_distances(self):
        self.prev_distances = {}
        for red in self.bf.red_uav:
            if not red.alive:
                continue
            target = self._current_target(red)
            if target is not None and target.alive:
                self.prev_distances[f"red_{red.id}"] = float(np.linalg.norm(target.pos - red.pos))

    def _current_target(self, red):
        agent_id = f"red_{red.id}"
        target_id = self.assigned_targets.get(agent_id)
        if target_id is not None and 0 <= int(target_id) < len(self.bf.blue_uav):
            target = self.bf.blue_uav[int(target_id)]
            if target.alive:
                return target
        candidates = select_candidate_targets(red, self.bf, top_k=1)
        return candidates[0] if candidates else None

    def _assign_targets(self):
        for red in self.bf.red_uav:
            agent_id = f"red_{red.id}"
            target = self._current_target(red) if agent_id in self.assigned_targets else None
            if target is None:
                candidates = select_candidate_targets(red, self.bf, top_k=1)
                target = candidates[0] if candidates else None
            self.assigned_targets[agent_id] = None if target is None else target.id

    def _record_low_level_actions(self, action_dict):
        action_counts = self.episode_stats.setdefault("low_action_counts", [0] * ACTION_COUNT)
        for action in action_dict.values():
            action_id = int(action)
            if 0 <= action_id < ACTION_COUNT:
                action_counts[action_id] += 1

    def _build_rewards(self):
        start_red_alive = self.prev_red_alive
        start_blue_alive = self.prev_blue_alive
        end_red_alive = self.bf.num_red_alive
        end_blue_alive = self.bf.num_blue_alive
        rewards = {}

        for red in self.bf.red_uav:
            agent_id = f"red_{red.id}"
            old_status = self.prev_status.get(agent_id, {"alive": False, "info": None})
            just_died = old_status["alive"] and not red.alive
            reward = 0.0

            if self.mode == "attack":
                reward += self._attack_reward(red, agent_id)
            else:
                reward += self._evade_reward(red, agent_id)

            if just_died:
                self.episode_stats["death_events"] += 1
                reward += (
                    LowLevelConfig["reward_attack_dead"]
                    if self.mode == "attack"
                    else LowLevelConfig["reward_evade_dead"]
                )
                if red.info == "out":
                    self.episode_stats["out_events"] += 1
                    reward += (
                        LowLevelConfig["reward_attack_out"]
                        if self.mode == "attack"
                        else LowLevelConfig["reward_evade_out"]
                    )

            rewards[agent_id] = float(reward)

        blue_losses = max(0, start_blue_alive - end_blue_alive)
        if blue_losses:
            self.episode_stats["kill_events"] += blue_losses

        if self.mode == "attack" and self.bf.battlefield_done["__all__"]:
            team_reward = 0.0
            if self.bf.battlefield_info == "red":
                team_reward += LowLevelConfig["reward_attack_win"]
            elif self.bf.battlefield_info == "blue":
                team_reward += LowLevelConfig["reward_attack_lose"]
            if start_red_alive > end_red_alive:
                team_reward += (start_red_alive - end_red_alive) * LowLevelConfig["reward_attack_dead"]
            if start_blue_alive > end_blue_alive:
                team_reward += (start_blue_alive - end_blue_alive) * LowLevelConfig["reward_attack_kill"]
            for agent_id in rewards:
                rewards[agent_id] += team_reward

        return rewards

    def _attack_reward(self, red, agent_id):
        if not red.alive:
            return 0.0
        target = self._current_target(red)
        reward = 0.0
        reward += LowLevelConfig["reward_attack_advantage"] * get_uav_advantage(red, target)
        if target is not None and target.alive:
            dist = np.linalg.norm(target.pos - red.pos)
            reward += np.exp(-max(0.0, dist - 1000.0) / 3000.0)

        red_record = self.bf.record["Red"][red.id]
        reward += LowLevelConfig["reward_attack_damage"] * red_record["damage"]
        reward += LowLevelConfig["reward_attack_hurt"] * red_record["hurt"]
        if red_record["damage"] > 0.0 and target is not None and not target.alive:
            reward += LowLevelConfig["reward_attack_kill"]
        return reward

    def _evade_reward(self, red, agent_id):
        reward = 0.0
        if red.alive:
            reward += LowLevelConfig["reward_evade_survive"]
        target = self._current_target(red)
        if target is None:
            return reward
        current_dist = float(np.linalg.norm(target.pos - red.pos))
        prev_dist = self.prev_distances.get(agent_id, current_dist)
        reward += LowLevelConfig["reward_evade_distance_scale"] * ((current_dist - prev_dist) / 16000.0)
        if red.alive:
            threat_adv = get_uav_advantage(target, red)
            reward += LowLevelConfig["reward_evade_threat_penalty"] * max(threat_adv, 0.0)
        return reward

    def call_back_winner(self):
        return self.bf.battlefield_info, self.bf.num_red_alive, self.bf.num_blue_alive

    def call_back_low_level_stats(self):
        stats = dict(self.episode_stats)
        action_counts = stats.get("low_action_counts", [0] * ACTION_COUNT)
        total_actions = sum(action_counts)
        for action_id, count in enumerate(action_counts):
            stats[f"low_action_{action_id:02d}_ratio"] = count / total_actions if total_actions else 0.0
            stats[f"low_action_{action_id:02d}_count"] = count
        return stats

    def call_back_save_data(self):
        self.bf.save_data()
