import numpy as np

from exp_article_test.battlefield import BattleField
from exp_article_test.config import HRLConfig
from exp_article_test.maneuver import build_low_level_action, make_decision_for_blue_team, select_candidate_targets
from exp_article_test.low_level_env import LowLevelController
from exp_article_test.toolkit import get_rela_angle


class BattleScene(object):
    def __init__(
        self,
        num_red,
        num_blue,
        candidate_enemy_count=None,
        sub_steps=None,
        use_blue_script=True,
        use_script_low_level=None,
        low_level_policy_dir=None,
        freeze_low_level=None,
    ):
        self.bf = BattleField(num_red, num_blue)
        self.candidate_enemy_count = candidate_enemy_count or HRLConfig["candidate_enemy_count"]
        self.sub_steps = sub_steps or HRLConfig["sub_steps"]
        self.use_blue_script = use_blue_script
        self.use_script_low_level = (
            HRLConfig["use_script_low_level"] if use_script_low_level is None else use_script_low_level
        )
        self.low_level_policy_dir = low_level_policy_dir or HRLConfig["low_level_policy_dir"]
        self.freeze_low_level = HRLConfig["freeze_low_level"] if freeze_low_level is None else freeze_low_level

        self.self_obs_dim = 8
        self.enemy_obs_dim = 11
        self.ally_obs_dim = 8
        self.high_level_obs_dim = (
            self.self_obs_dim
            + self.candidate_enemy_count * self.enemy_obs_dim
            + self.ally_obs_dim
        )
        self.action_dim = self.candidate_enemy_count + 1

        self.last_rewards = {}
        self.step_stats = {"attack_count": 0, "evade_count": 0}
        self.episode_stats = {"attack_count": 0, "evade_count": 0}
        self.low_level_controller = LowLevelController(
            use_script=self.use_script_low_level,
            policy_dir=self.low_level_policy_dir,
            device="cpu",
        )

    def reset(self):
        self.bf.reset()
        self.last_rewards = {
            f"red_{uav.id}": 0.0
            for uav in self.bf.red_uav
            if uav.alive
        }
        self.step_stats = {"attack_count": 0, "evade_count": 0}
        self.episode_stats = {"attack_count": 0, "evade_count": 0}

    def step(self, macro_actions):
        self.step_stats = {"attack_count": 0, "evade_count": 0}
        start_red_alive = self.bf.num_red_alive
        start_blue_alive = self.bf.num_blue_alive
        prev_status = {
            f"red_{uav.id}": {"alive": uav.alive, "info": uav.info}
            for uav in self.bf.red_uav
        }

        shaped_rewards = self._action_assessment(macro_actions)
        executed_terms = {}

        for _ in range(self.sub_steps):
            red_actions = {}
            blue_actions = make_decision_for_blue_team(self.bf) if self.use_blue_script else {}
            executed_terms = {}

            for red in self.bf.red_uav:
                if not red.alive:
                    continue
                agent_id = f"red_{red.id}"
                macro_action = int(macro_actions.get(agent_id, 0))
                mode = 0 if macro_action <= 0 else 1
                target_idx = max(0, macro_action - 1)
                low_level_action, target = self.infer_low_level_action(
                    mode=mode,
                    target_idx=target_idx,
                    uav=red,
                )
                red_actions[agent_id] = int(low_level_action)
                executed_terms[agent_id] = {
                    "macro_mode": mode,
                    "macro_target": None if target is None else target.id,
                }

                if mode == 0:
                    self.step_stats["evade_count"] += 1
                    self.episode_stats["evade_count"] += 1
                else:
                    self.step_stats["attack_count"] += 1
                    self.episode_stats["attack_count"] += 1

                red_record = self.bf.record["Red"][red.id]
                red_record["macro_mode"] = mode
                red_record["macro_target"] = None if target is None else target.id

            if red_actions:
                self.bf.step(red_actions, blue_actions)
            else:
                self.bf.check_winner()

            if self.bf.battlefield_done["__all__"]:
                break

        self.last_rewards = self._build_rewards(
            macro_actions=macro_actions,
            shaped_rewards=shaped_rewards,
            prev_status=prev_status,
            start_red_alive=start_red_alive,
            start_blue_alive=start_blue_alive,
            executed_terms=executed_terms,
        )

    def observation(self):
        obs = {}
        for uav in self.bf.red_uav:
            agent_id = f"red_{uav.id}"
            if not self.bf.battlefield_done[agent_id]:
                obs[agent_id] = self.get_high_level_obs(uav)
        return obs

    def reward(self):
        return {agent_id: float(value) for agent_id, value in self.last_rewards.items()}

    def done(self):
        return self.bf.battlefield_done

    def info(self):
        info = {}
        total_step_actions = self.step_stats["attack_count"] + self.step_stats["evade_count"]
        total_episode_actions = self.episode_stats["attack_count"] + self.episode_stats["evade_count"]
        for uav in self.bf.red_uav:
            agent_id = f"red_{uav.id}"
            if not self.bf.battlefield_done[agent_id]:
                info[agent_id] = {
                    "step_attack_ratio": (
                        self.step_stats["attack_count"] / total_step_actions if total_step_actions else 0.0
                    ),
                    "step_evade_ratio": (
                        self.step_stats["evade_count"] / total_step_actions if total_step_actions else 0.0
                    ),
                    "episode_attack_ratio": (
                        self.episode_stats["attack_count"] / total_episode_actions if total_episode_actions else 0.0
                    ),
                    "episode_evade_ratio": (
                        self.episode_stats["evade_count"] / total_episode_actions if total_episode_actions else 0.0
                    ),
                }
        return info

    def get_high_level_obs(self, uav):
        if not uav.alive:
            return np.zeros((self.high_level_obs_dim,), dtype=np.float32)

        obs_parts = [self._self_features(uav)]
        candidates = select_candidate_targets(uav, self.bf, top_k=self.candidate_enemy_count)
        for idx in range(self.candidate_enemy_count):
            if idx < len(candidates):
                obs_parts.append(self._enemy_features(uav, candidates[idx]))
            else:
                obs_parts.append(np.zeros(self.enemy_obs_dim, dtype=np.float32))
        obs_parts.append(self._ally_features(uav))
        return np.concatenate(obs_parts).astype(np.float32)

    def _self_features(self, uav):
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

    def _enemy_features(self, uav, enemy):
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

    def _ally_features(self, uav):
        allies = []
        for ally in self.bf.red_uav:
            if ally.id == uav.id or not ally.alive:
                continue
            allies.append((ally, np.linalg.norm(ally.pos - uav.pos)))
        if not allies:
            return np.zeros(self.ally_obs_dim, dtype=np.float32)
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

    def _get_uav_advantage(self, uav, enemy):
        rel = enemy.pos - uav.pos
        dist = np.linalg.norm(rel)
        my_angle = get_rela_angle(uav.vel, rel)
        enemy_angle = get_rela_angle(enemy.vel, -rel)
        dist_score = np.exp(-max(0.0, dist - 1000.0) / 3000.0)
        angle_score = 1.0 - (my_angle / np.pi)
        pressure = 1.0 - (enemy_angle / np.pi)
        return angle_score * dist_score - pressure

    def _action_assessment(self, macro_actions):
        shaped_rewards = {}
        reward_scale = HRLConfig["reward_action_assess"]
        for red in self.bf.red_uav:
            agent_id = f"red_{red.id}"
            if not red.alive:
                shaped_rewards[agent_id] = 0.0
                continue

            macro_action = int(macro_actions.get(agent_id, 0))
            candidates = select_candidate_targets(red, self.bf, top_k=self.candidate_enemy_count)
            if not candidates:
                shaped_rewards[agent_id] = 0.0
                continue

            nearest_enemy = candidates[0]
            threat_score = self._get_uav_advantage(red, nearest_enemy)

            if macro_action <= 0:
                shaped_rewards[agent_id] = reward_scale if threat_score < -0.05 else 0.0
            else:
                target_idx = min(macro_action - 1, len(candidates) - 1)
                target = candidates[target_idx]
                attack_score = self._get_uav_advantage(red, target)
                shaped_rewards[agent_id] = reward_scale if attack_score > 0.0 else 0.0
        return shaped_rewards

    def infer_low_level_action(self, mode, target_idx, uav):
        if self.use_script_low_level:
            return build_low_level_action(
                mode=mode,
                target_idx=target_idx,
                uav=uav,
                bf=self.bf,
                top_k=self.candidate_enemy_count,
            )

        candidates = select_candidate_targets(uav, self.bf, top_k=self.candidate_enemy_count)
        if mode <= 0:
            action = self.low_level_controller.infer_action("evade", uav, self.bf, target=None)
            return action, None

        index = min(max(target_idx, 0), len(candidates) - 1) if candidates else 0
        target = candidates[index] if candidates else None
        if target is None:
            action = self.low_level_controller.infer_action("evade", uav, self.bf, target=None)
            return action, None
        action = self.low_level_controller.infer_action("attack", uav, self.bf, target=target)
        return action, target

    def _build_rewards(self, macro_actions, shaped_rewards, prev_status, start_red_alive, start_blue_alive, executed_terms):
        rewards = {}
        end_red_alive = self.bf.num_red_alive
        end_blue_alive = self.bf.num_blue_alive
        team_reward = (
            (start_blue_alive - end_blue_alive) * HRLConfig["reward_kill"]
            + (start_red_alive - end_red_alive) * HRLConfig["reward_dead"]
        )

        if self.bf.battlefield_done["__all__"]:
            if self.bf.battlefield_info == "red":
                team_reward += HRLConfig["reward_win"]
            elif self.bf.battlefield_info == "blue":
                team_reward += HRLConfig["reward_lose"]

        for red in self.bf.red_uav:
            agent_id = f"red_{red.id}"
            reward = shaped_rewards.get(agent_id, 0.0) + team_reward
            old_status = prev_status.get(agent_id, {"alive": False, "info": None})
            just_died = old_status["alive"] and not red.alive
            if just_died and red.info == "out":
                reward += HRLConfig["reward_out"]

            reward_terms = {
                "team_reward": team_reward,
                "shaped_reward": shaped_rewards.get(agent_id, 0.0),
                "just_died": just_died,
                "macro_mode": executed_terms.get(agent_id, {}).get("macro_mode"),
                "macro_target": executed_terms.get(agent_id, {}).get("macro_target"),
            }
            self.bf.record["Red"][red.id]["last_macro_reward_terms"] = reward_terms

            rewards[agent_id] = float(reward)
        return rewards

    def get_episode_stats(self):
        total_actions = self.episode_stats["attack_count"] + self.episode_stats["evade_count"]
        return {
            "attack_ratio": self.episode_stats["attack_count"] / total_actions if total_actions else 0.0,
            "evade_ratio": self.episode_stats["evade_count"] / total_actions if total_actions else 0.0,
            "attack_count": self.episode_stats["attack_count"],
            "evade_count": self.episode_stats["evade_count"],
        }

    def call_back_winner(self):
        return self.bf.battlefield_info, self.bf.num_red_alive, self.bf.num_blue_alive

    def call_back_save_data(self):
        self.bf.save_data()
