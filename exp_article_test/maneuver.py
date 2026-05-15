import numpy as np

from exp_article_test.config import HRLConfig, UAVConfig
from exp_article_test.toolkit import get_rela_angle


def convert_to_contol_input(rl_action):
    if rl_action == 0:
        return np.array([0.0, 1.0, 0.0])
    if rl_action == 1:
        return np.array([4.0, 1.0, 0.0])
    if rl_action == 2:
        return np.array([-4.0, 1.0, 0.0])
    if rl_action == 3:
        return np.array([0.0, 7.0, 0.0])
    if rl_action == 4:
        return np.array([4.0, 7.0, 0.0])
    if rl_action == 5:
        return np.array([-4.0, 7.0, 0.0])
    if rl_action == 6:
        return np.array([0.0, -5.0, 0.0])
    if rl_action == 7:
        return np.array([4.0, -5.0, 0.0])
    if rl_action == 8:
        return np.array([-4.0, -5.0, 0.0])
    if rl_action == 9:
        return np.array([0.0, 7.0, np.arccos(2 / 7)])
    if rl_action == 10:
        return np.array([4.0, 7.0, np.arccos(2 / 7)])
    if rl_action == 11:
        return np.array([-4.0, 7.0, np.arccos(2 / 7)])
    if rl_action == 12:
        return np.array([0.0, 7.0, -np.arccos(2 / 7)])
    if rl_action == 13:
        return np.array([4.0, 7.0, -np.arccos(2 / 7)])
    if rl_action == 14:
        return np.array([-4.0, 7.0, -np.arccos(2 / 7)])
    raise ValueError(f"Unsupported discrete action: {rl_action}")


def make_action_array():
    return [convert_to_contol_input(i) for i in range(15)]


def estimate_status(uav, action):
    nx = action[0]
    nn = action[1]
    phi = action[2]
    c_the = np.cos(uav.theta)
    s_the = np.sin(uav.theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    pos = np.array(uav.pos) + np.array(uav.vel) * uav.dt
    airspeed = uav.airspeed + 9.8 * (nx - s_the) * uav.dt
    airspeed = np.clip(airspeed, 100.0, 250.0)
    theta = uav.theta + 9.8 * (nn * c_phi - c_the) / (airspeed + 1e-6) * uav.dt
    psi = uav.psi + 9.8 * nn * s_phi / (airspeed * c_the + 1e-6) * uav.dt
    c_the = np.cos(theta)
    s_the = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    vel = np.array([airspeed * c_the * c_psi, airspeed * c_the * s_psi, -airspeed * s_the])
    return np.array([pos, vel])


ATTACK_CONE_ANGLE = np.pi / 6
MAX_NX = UAVConfig.get("nx_max", 4.0)
MAX_NN = UAVConfig.get("nn_max", 7.0)
BASE_BANK = np.arccos(2.0 / 7.0)


def _desired_heading_to_position(uav, position):
    rel = position - uav.pos
    psi_d = np.arctan2(rel[1], rel[0])
    horiz = np.hypot(rel[0], rel[1])
    theta_d = -np.arctan2(rel[2], horiz)
    return theta_d, psi_d


def _steer_towards_heading(uav, theta_d, psi_d, speed_mode="attack"):
    d_psi = np.arctan2(np.sin(psi_d - uav.psi), np.cos(psi_d - uav.psi))
    d_the = np.arctan2(np.sin(theta_d - uav.theta), np.cos(theta_d - uav.theta))

    nx = float(MAX_NX if speed_mode == "attack" else 3.0)
    nn = float(MAX_NN)
    phi = BASE_BANK * (1.0 if d_psi >= 0.0 else -1.0)

    if abs(d_the) > ATTACK_CONE_ANGLE / 2.0:
        phi = 0.0
        nn = float(MAX_NN) * (1.0 if d_the > 0.0 else -1.0)

    if speed_mode == "evade":
        nx = float(MAX_NX)
        nn = float(MAX_NN)
        if abs(d_psi) < np.deg2rad(8.0):
            phi = BASE_BANK if d_psi >= 0.0 else -BASE_BANK

    nn = np.clip(nn, UAVConfig["nn_min"], UAVConfig["nn_max"])
    nx = np.clip(nx, UAVConfig["nx_min"], UAVConfig["nx_max"])
    return np.array([nx, nn, phi], dtype=float)


def pursuit_script_action(uav, target):
    theta_d, psi_d = _desired_heading_to_position(uav, target.pos)
    action = _steer_towards_heading(uav, theta_d, psi_d, speed_mode="attack")

    rel = target.pos - uav.pos
    if np.linalg.norm(rel) > 1e-6:
        rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
        my_dir = uav.vel / (np.linalg.norm(uav.vel) + 1e-6)
        ang = np.arccos(np.clip(np.dot(rel_dir, my_dir), -1.0, 1.0))
        if ang <= ATTACK_CONE_ANGLE:
            action[2] = 0.0
            action[1] = 0.5 * MAX_NN
    return action


def evade_script_action(uav, threat):
    rel = threat.pos - uav.pos
    away_pos = uav.pos - rel
    away_pos[2] = np.clip(uav.pos[2] + np.sign(rel[2] + 1e-6) * 300.0, -7500.0, -500.0)
    theta_d, psi_d = _desired_heading_to_position(uav, away_pos)
    action = _steer_towards_heading(uav, theta_d, psi_d, speed_mode="evade")

    rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
    my_dir = uav.vel / (np.linalg.norm(uav.vel) + 1e-6)
    if np.dot(rel_dir, my_dir) > 0.2:
        action[2] = BASE_BANK if action[2] >= 0.0 else -BASE_BANK
        action[1] = MAX_NN
    return action


def select_discrete_action(cont_action):
    action_array = make_action_array()
    cont_action = np.array(cont_action)
    distances = [np.linalg.norm(cont_action - a) for a in action_array]
    return int(np.argmin(distances))


def matrix_game(uav, target):
    uav_action_array = make_action_array()
    target_action_array = make_action_array()
    game_mat = np.zeros(shape=(len(uav_action_array), len(target_action_array)))
    uav_status = []
    target_status = []
    for uav_action in uav_action_array:
        uav_status.append(estimate_status(uav, uav_action))
    for target_action in target_action_array:
        target_status.append(estimate_status(target, target_action))
    for i, uav_stat in enumerate(uav_status):
        for j, target_stat in enumerate(target_status):
            rel_pos = target_stat[0] - uav_stat[0]
            dis = np.linalg.norm(rel_pos)
            theta_r = get_rela_angle(rel_pos, uav_stat[1])
            theta_b = get_rela_angle(rel_pos, target_stat[1])
            angle_adv = 1.0 - (theta_r + theta_b) / np.pi
            dis_adv = np.exp(-max(0.0, dis - 1000.0) / 1000.0)
            game_mat[i][j] = angle_adv * dis_adv
    select_arr = np.min(game_mat, axis=1)
    select_action_idx = int(np.argmax(select_arr))
    return uav_action_array[select_action_idx]


def _sorted_alive_targets(uav, target_team):
    candidates = []
    for target in target_team:
        if not target.alive:
            continue
        dis = np.linalg.norm(uav.pos - target.pos)
        candidates.append((target, dis))
    candidates.sort(key=lambda item: item[1])
    return candidates


def select_candidate_targets(uav, bf, top_k=None):
    if top_k is None:
        top_k = HRLConfig["candidate_enemy_count"]
    target_team = bf.blue_uav if uav.team == "Red" else bf.red_uav
    candidates = _sorted_alive_targets(uav, target_team)
    return [target for target, _ in candidates[:top_k]]


def option_attack(uav, target, bf):
    del bf
    if target is None or not target.alive:
        return None
    cont_action = pursuit_script_action(uav, target)
    return select_discrete_action(cont_action)


def option_evade(uav, threats, bf):
    del bf
    if not threats:
        return select_discrete_action(np.array([MAX_NX, 1.0, BASE_BANK], dtype=float))
    threat = threats[0]
    cont_action = evade_script_action(uav, threat)
    return select_discrete_action(cont_action)


def build_low_level_action(mode, target_idx, uav, bf, top_k=None):
    candidates = select_candidate_targets(uav, bf, top_k=top_k)
    if mode <= 0:
        return option_evade(uav, candidates, bf), None

    index = max(0, target_idx)
    if index >= len(candidates):
        index = 0
    target = candidates[index] if candidates else None
    if target is None:
        return option_evade(uav, candidates, bf), None
    return option_attack(uav, target, bf), target


def lock_red_target(uav, bf):
    candidates = select_candidate_targets(uav, bf, top_k=1)
    return candidates[0].id if candidates else -1


def lock_blue_target(uav, bf):
    candidates = select_candidate_targets(uav, bf, top_k=1)
    return candidates[0].id if candidates else -1


def make_decision_for_blue_team(bf):
    blue_actions = {}
    for blue in bf.blue_uav:
        if not blue.alive:
            continue
        candidates = select_candidate_targets(blue, bf, top_k=1)
        if not candidates:
            blue_actions[f"blue_{blue.id}"] = 0
            continue
        cont_action = pursuit_script_action(blue, candidates[0])
        blue_actions[f"blue_{blue.id}"] = select_discrete_action(cont_action)
    return blue_actions


def make_decision_for_red_team(bf):
    red_actions = {}
    for red in bf.red_uav:
        if not red.alive:
            continue
        candidates = select_candidate_targets(red, bf, top_k=1)
        if not candidates:
            red_actions[f"red_{red.id}"] = 0
            continue
        cont_action = pursuit_script_action(red, candidates[0])
        red_actions[f"red_{red.id}"] = select_discrete_action(cont_action)
    return red_actions
