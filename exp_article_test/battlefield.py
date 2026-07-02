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
        self.boundary_x_min = self.config["boundary_x_min"]
        self.boundary_x_max = self.config["boundary_x_max"]
        self.boundary_y_min = self.config["boundary_y_min"]
        self.boundary_y_max = self.config["boundary_y_max"]
        self.boundary_z_min = self.config["boundary_z_min"]
        self.boundary_z_max = self.config["boundary_z_max"]

        self.record = {}
        self.last_record = {}
        self.battlefield_done = {}
        self.battlefield_info = "continue"

    def reset(self):
        self.t = 0
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

    def step(self, red_actions, blue_actions):
        self.last_record = deepcopy(self.record)
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
            self.check_after_step()
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

    def check_after_step(self):
        self.check_out()
        self.check_crash()
        for uav in itertools.chain(self.red_uav, self.blue_uav):
            if not uav.alive:
                continue
            self.check_damage(uav)
        for uav in itertools.chain(self.red_uav, self.blue_uav):
            if uav.alive:
                uav.record_status()

    # def check_damage(self, uav):
    #     current_damage = 0.0
    #     target_id = None
    #     if uav.team == "Red":
    #         target_team = self.blue_uav
    #     else:
    #         target_team = self.red_uav
    #     for target in target_team:
    #         if not target.alive:
    #             continue
    #         damage = self.cal_damage(uav, target)
    #         if damage > current_damage:
    #             current_damage = damage
    #     if target_id is not None:
    #         uav.set_target(target_id)
    #         target = target_team[int(target_id)]
    #         if uav.team == "Red":
    #             current_damage *= 2.5
    #         else:
    #             current_damage *= 0.8
    #         target.blood -= current_damage
    #         if target.blood < 0.0:
    #             target.destroy("hit")
    #         self.record[uav.team][uav.id]['target_blood'] = target.blood
    #         self.record[uav.team][uav.id]['damage'] = current_damage
    #         self.record[target.team][target.id]['hurt'] = current_damage
#baixiaoxiao
    def check_damage(self, uav):
        # 1) 选出当前时刻最该打的人（伤害最大的那个）
        best_damage = 0.0
        best_target = None

        target_team = self.blue_uav if uav.team == "Red" else self.red_uav
        for target in target_team:
            if not target.alive:
                continue
            d = self.cal_damage(uav, target)
            if d > best_damage:
                best_damage = d
                best_target = target  # ← 关键：把目标记下来

        # 2) 没有命中任何目标就退出，并把本机当步伤害清零（避免残留旧值）
        if best_target is None or best_damage <= 0.0:
            self.record[uav.team][uav.id]['damage'] = 0.0
            return

        # 3) 伤害系数。当前保持红蓝 1:1，避免胜率受阵营伤害倍率影响。
        if uav.team == "Red":
            best_damage *= 1.0
        else:
            best_damage *= 1.0

        # 4) 结算到被打的那架机上
        uav.set_target(best_target.id)
        best_target.blood -= best_damage
        if best_target.blood <= 0.0:
            best_target.destroy("hit")

        # 5) 记录双方账本（供奖励函数使用）
        self.record[uav.team][uav.id]['target_blood'] = max(0.0, best_target.blood)
        self.record[uav.team][uav.id]['damage'] = best_damage
        self.record[best_target.team][best_target.id]['hurt'] = best_damage

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

