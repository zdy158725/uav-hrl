import itertools
from copy import deepcopy

from exp_article_test.maneuver import convert_to_contol_input
from exp_article_test.uav import UAV
from exp_article_test.config import BattleFieldConfig, PathConfig
from exp_article_test.toolkit import *

class BattleField:
    def __init__(self, num_red, num_blue):
        self.t = 0
        self.dt = 0.1
        self.num_red = num_red
        self.num_blue = num_blue
        self.num_red_alive = num_red
        self.num_blue_alive = num_blue
        self.red_uav = []
        self.blue_uav = []

        for i in range(self.num_red):
            self.red_uav.append(UAV("Red", str(i)))
        for i in range(self.num_blue):
            self.blue_uav.append(UAV("Blue", str(i)))

        self.config = BattleFieldConfig
        self.trajectory_path = PathConfig["trajectory_path"]
        self.max_simulation_time = self.config["max_simulation_time"]
        self.decision_sep = self.config["decision_separation"]
        self.crash_distance = self.config["crash_distance"]
        self.fire_cooldown_seconds = float(self.config["fire_cooldown_seconds"])
        if self.fire_cooldown_seconds < 0.0:
            raise ValueError("fire_cooldown_seconds must be non-negative")
        self.evade_damage_multiplier = float(self.config["evade_damage_multiplier"])
        if not 0.0 <= self.evade_damage_multiplier <= 1.0:
            raise ValueError("evade_damage_multiplier must be between 0 and 1")
        self.boundary_x_min = self.config["boundary_x_min"]
        self.boundary_x_max = self.config["boundary_x_max"]
        self.boundary_y_min = self.config["boundary_y_min"]
        self.boundary_y_max = self.config["boundary_y_max"]
        self.boundary_z_min = self.config["boundary_z_min"]
        self.boundary_z_max = self.config["boundary_z_max"]

        self.record = {}
        self.last_record = {}
        self.last_damage_time = {}
        self.battlefield_done = {}
        self.battlefield_info = "continue"

    def reset(self):
        self.t = 0
        self.last_damage_time = {}
        self.reset_record()
        self.num_red_alive = self.num_red
        self.num_blue_alive = self.num_blue

        pos = self.get_init_pos()
        for i, red in enumerate(self.red_uav):
            red.reset(pos=pos['Red'][i], airspeed=200, psi=0)
        for i, blue in enumerate(self.blue_uav):
            blue.reset(pos=pos['Blue'][i], airspeed=200, psi=np.pi)

        self.check_winner()

    def generate_coordinates(self, count, x_min, x_max, y_min, y_max, z_min, z_max):
        """
        满足距离要求的随机坐标列表
        返回:
            包含坐标的列表，每个坐标是[x, y, z]三元组
        """
        points = []
        while len(points) < count:
            # 生成新候选点
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = np.random.uniform(z_min, z_max)
            new_point = np.array([x, y, z])
            # 如果是第一个点，直接添加
            if not points:
                points.append(new_point)
                continue
            # 检查新点与所有已有点的距离
            valid = True
            for existing in points:
                distance = np.linalg.norm(new_point - existing)
                if distance <= self.crash_distance:
                    valid = False
                    break
            # 如果满足距离要求，添加到列表
            if valid:
                points.append(new_point)
        return points

    def get_init_pos(self):
        pos = {'Red': self.generate_coordinates(self.num_red, 5000, 7000, 7000, 9000, -5000, -4000),
               'Blue': self.generate_coordinates(self.num_blue, 9000, 11000, 7000, 9000, -5000, -4000)}
        return pos

    def step(self, red_actions, blue_actions, *, attack_agent_ids, attack_targets):
        self.last_record = deepcopy(self.record)
        attack_agent_ids = frozenset(attack_agent_ids)
        attack_targets = dict(attack_targets)
        for uav in itertools.chain(self.red_uav, self.blue_uav):
            self.record[uav.team][uav.id]['damage'] = 0.0
            self.record[uav.team][uav.id]['hurt'] = 0.0
        for _ in range(self.decision_sep):
            self.t += self.dt
            for red in self.red_uav:
                if not red.alive:
                    continue
                action_key = f"red_{red.id}"
                if action_key not in red_actions:
                    raise KeyError(f"Missing action for alive red UAV: {action_key}")
                action_index = int(red_actions[action_key])
                self.record["Red"][red.id]["action_index"] = action_index
                real_action = convert_to_contol_input(action_index)
                red.step(real_action)
            for blue in self.blue_uav:
                if not blue.alive:
                    continue
                action_key = f"blue_{blue.id}"
                if action_key not in blue_actions:
                    raise KeyError(f"Missing action for alive blue UAV: {action_key}")
                action_index = int(blue_actions[action_key])
                self.record["Blue"][blue.id]["action_index"] = action_index
                real_action = convert_to_contol_input(action_index)
                blue.step(real_action)
            self.check_after_step(attack_agent_ids, attack_targets)
        self.check_winner()

    def reset_record(self):
        self.record = {}
        self.last_record = {}
        self.record['Red'] = {}
        for red in self.red_uav:
            self.record['Red'][red.id] = {}
            self.record['Red'][red.id]['damage'] = 0.0
            self.record['Red'][red.id]['hurt'] = 0.0
            self.record['Red'][red.id]['hit'] = False
            self.record['Red'][red.id]['target_blood'] = 1.0
            self.record['Red'][red.id]['blood'] = 1.0
            self.record['Red'][red.id]['action_index'] = None
            self.record['Red'][red.id]['macro_mode'] = None
            self.record['Red'][red.id]['macro_target'] = None
            self.record['Red'][red.id]['last_macro_reward_terms'] = {}
        self.record['Blue'] = {}
        for blue in self.blue_uav:
            self.record['Blue'][blue.id] = {}
            self.record['Blue'][blue.id]['damage'] = 0.0
            self.record['Blue'][blue.id]['hurt'] = 0.0
            self.record['Blue'][blue.id]['hit'] = False
            self.record['Blue'][blue.id]['target_blood'] = 1.0
            self.record['Blue'][blue.id]['blood'] = 1.0
            self.record['Blue'][blue.id]['action_index'] = None
            self.record['Blue'][blue.id]['macro_mode'] = None
            self.record['Blue'][blue.id]['macro_target'] = None
            self.record['Blue'][blue.id]['last_macro_reward_terms'] = {}
        self.record['n_alive_red'] = self.num_red
        self.record['n_alive_blue'] = self.num_blue
        self.last_record = deepcopy(self.record)

    def check_after_step(self, attack_agent_ids, attack_targets):
        self.check_out()
        self.check_crash()
        alive_uavs = [uav for uav in itertools.chain(self.red_uav, self.blue_uav) if uav.alive]

        # Determine every attack from the same pre-damage state. A target that is
        # destroyed this tick must still be able to complete its own attack.
        pending_hits = []
        pending_damage = {}
        for uav in alive_uavs:
            agent_id = f"{uav.team.lower()}_{uav.id}"
            if agent_id not in attack_agent_ids:
                continue
            elapsed = self.t - self.last_damage_time.get(agent_id, float("-inf"))
            if elapsed + 1e-9 < self.fire_cooldown_seconds:
                continue
            target_id = attack_targets[agent_id]
            target, damage = self.check_damage(uav, attack_agent_ids, target_id)
            if target is None:
                continue
            pending_hits.append((uav, target, damage))
            pending_damage[target] = pending_damage.get(target, 0.0) + damage

        for target, damage in pending_damage.items():
            target.blood -= damage
        for attacker, target, damage in pending_hits:
            attacker.set_target(target.id)
            self.last_damage_time[f"{attacker.team.lower()}_{attacker.id}"] = self.t
            self.record[attacker.team][attacker.id]['damage'] += damage
            self.record[attacker.team][attacker.id]['target_blood'] = max(0.0, target.blood)
            self.record[target.team][target.id]['hurt'] += damage
        for target in pending_damage:
            if target.blood <= 0.0:
                target.destroy("hit")

        for uav in alive_uavs:
            if uav.alive:
                uav.record_status()

    def check_damage(self, uav, attack_agent_ids, target_id):
        """只计算指定敌机的伤害，不在遍历期间修改飞机状态。"""
        target_team = self.blue_uav if uav.team == "Red" else self.red_uav
        target = next((enemy for enemy in target_team if enemy.id == str(target_id) and enemy.alive), None)
        if target is None:
            return None, 0.0

        damage = self.cal_damage(uav, target)
        if f"{target.team.lower()}_{target.id}" not in attack_agent_ids:
            damage *= self.evade_damage_multiplier
        if damage <= 0.0:
            return None, 0.0
        return target, damage

    def cal_damage(self, uav, target):
        rel_pos = target.pos - uav.pos
        dis = np.linalg.norm(rel_pos)
        if dis > 1000.0:
            return 0.0
        if get_rela_angle(uav.vel, rel_pos) > np.pi / 6:
            return 0.0
        Dr = (1000.0 - dis) / 1000.0 * 0.3 + 0.7
        theta = get_rela_angle(uav.vel, target.vel)
        Dtheta = (np.exp(-np.linalg.norm(target.vel) * np.sin(theta) / (dis * np.tan(np.pi / 18.0) + 1e-6)) * 0.7
                  + 0.3)
        r_uav = np.cross(uav.vel, uav.acc)
        r_tar = np.cross(target.vel, target.acc)
        theta_c = np.arccos(
            np.clip(r_uav.dot(r_tar) / (np.linalg.norm(r_uav) * np.linalg.norm(r_tar) + 1e-6), -1.0, 1.0))
        Dc = (1.0 - np.sqrt(np.sin(theta_c))) * 0.9 + 0.1
        return Dr * Dtheta * Dc

    def check_crash(self):
        for id1, uav1 in enumerate(itertools.chain(self.red_uav, self.blue_uav)):
            if not uav1.alive:
                continue
            for id2, uav2 in enumerate(itertools.chain(self.red_uav, self.blue_uav)):
                if id2 <= id1:
                    continue
                if not uav2.alive:
                    continue
                if np.abs(uav1.pos[0] - uav2.pos[0]) <= self.crash_distance:
                    if np.abs(uav1.pos[1] - uav2.pos[1]) <= self.crash_distance:
                        if np.linalg.norm(uav1.pos - uav2.pos) <= self.crash_distance:
                            uav1.destroy("crash")
                            uav2.destroy("crash")

    def check_out(self):
        for uav in itertools.chain(self.red_uav, self.blue_uav):
            if not uav.alive:
                continue
            if uav.pos[0] < self.boundary_x_min or uav.pos[0] > self.boundary_x_max:
                uav.destroy("out")
            elif uav.pos[1] < self.boundary_y_min or uav.pos[1] > self.boundary_y_max:
                uav.destroy("out")
            elif -uav.pos[2] < self.boundary_z_min or -uav.pos[2] > self.boundary_z_max:
                uav.destroy("out")

    def check_winner(self):
        self.num_red_alive = 0
        self.num_blue_alive = 0
        self.battlefield_done = {}
        for uav in self.red_uav:
            agent_id = f"red_{uav.id}"
            if uav.alive:
                self.num_red_alive += 1
                self.battlefield_done[agent_id] = False
            else:
                self.battlefield_done[agent_id] = True
        for uav in self.blue_uav:
            if uav.alive:
                self.num_blue_alive += 1
        self.record['n_alive_red'] = self.num_red_alive
        self.record['n_alive_blue'] = self.num_blue_alive
        if self.num_red_alive == 0 and self.num_blue_alive == 0:
            self.battlefield_done['__all__'] = True
            self.battlefield_info = "draw"
        elif self.num_red_alive == 0:
            self.battlefield_done['__all__'] = True
            self.battlefield_info = "blue"
        elif self.num_blue_alive == 0:
            self.battlefield_done['__all__'] = True
            self.battlefield_info = "red"
        elif self.t > self.max_simulation_time:
            self.battlefield_done['__all__'] = True
            self.battlefield_info = "timeout"
        else:
            self.battlefield_done['__all__'] = False
            self.battlefield_info = "continue"

    def save_data(self):
        current_time = get_time_string()
        filepath = self.trajectory_path + "/" + current_time + self.battlefield_info + "/"
        create_directory(filepath, [""])
        for uav in itertools.chain(self.red_uav, self.blue_uav):
            save_uav_csv(uav, filepath)

