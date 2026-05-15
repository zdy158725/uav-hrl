from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

UAVConfig = {
    "airspeed_min": 100,
    "airspeed_max": 280,
    "nn_min": -5,
    "nn_max": 7,
    "nx_min": -4,
    "nx_max": 4,
    "dphi_min": -np.pi * 4 / 3,
    "dphi_max": np.pi * 4 / 3,
    "D": 1.0,
    "Dr": 0.7,
    "Dt": 0.3,
}

BattleFieldConfig = {
    "max_simulation_time": 500.0,
    "decision_separation": 10,
    "crash_distance": 15.0,
    "boundary_x_min": 0,
    "boundary_x_max": 16000,
    "boundary_y_min": 0,
    "boundary_y_max": 16000,
    "boundary_z_min": 500,
    "boundary_z_max": 8000,
}

PathConfig = {
    "checkpoint_path": str(OUTPUT_DIR / "checkpoints"),
    "trajectory_path": str(OUTPUT_DIR / "trajectories"),
    "low_level_path": str(OUTPUT_DIR / "low_level"),
    "policy_path": str(OUTPUT_DIR / "policies"),
}

HRLConfig = {
    "candidate_enemy_count": 2,
    "sub_steps": 10,
    "high_level_mode_count": 3,
    "reward_kill": 10.0,
    "reward_dead": -10.0,
    "reward_win": 30.0,
    "reward_lose": -30.0,
    "reward_out": -5.0,
    "reward_action_assess": 0.2,
    "start_train_num_red": 2,
    "start_train_num_blue": 2,
    "use_script_low_level": True,
    "low_level_policy_dir": str(OUTPUT_DIR / "policies"),
    "freeze_low_level": True,
}

LowLevelConfig = {
    "threat_count": 2,
    "attack_obs_dim": 27,
    "evade_obs_dim": 38,
    "reward_attack_kill": 10.0,
    "reward_attack_advantage": 1.5,
    "reward_attack_damage": 2.0,
    "reward_attack_hurt": -1.0,
    "reward_attack_dead": -8.0,
    "reward_attack_out": -5.0,
    "reward_attack_win": 20.0,
    "reward_attack_lose": -20.0,
    "reward_evade_survive": 0.2,
    "reward_evade_distance_scale": 5.0,
    "reward_evade_threat_penalty": -0.6,
    "reward_evade_dead": -10.0,
    "reward_evade_out": -5.0,
    "start_train_num_red": 2,
    "start_train_num_blue": 2,
}

RayCheckpointConfig = {
    "training_iteration": 1000,
    "num_to_keep": 5,
    "checkpoint_frequency": 20,
}

RayTrainingConfig = {
    "learning_rate": 1e-4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 0.1,
}

RayModelConfig = {
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
    }
}
