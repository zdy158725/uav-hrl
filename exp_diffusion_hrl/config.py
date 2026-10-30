from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DiffusionHRLConfig = {
    "history_steps": 30,
    "prediction_steps": (10, 20, 30),
    "num_inference_steps": 10,
    "candidate_enemy_count": 2,
    "high_level_obs_dim": 58,
    "num_red": 3,
    "num_blue": 3,
    "sub_steps": 2,
    "prediction_device": "cuda",
    "predictor_bundle": str(PROJECT_ROOT / "outputs/diffusion_hrl/predictor"),
    "baseline_bundle": str(
        PROJECT_ROOT / "outputs/baselines/hrl_3v3_selfplay_ckpt220_lowppo_v1"
    ),
    "output_dir": str(PROJECT_ROOT / "outputs/diffusion_hrl"),
    "training_iterations": 80,
    "checkpoint_frequency": 20,
}
