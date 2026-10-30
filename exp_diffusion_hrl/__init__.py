"""Diffusion trajectory prediction augmented hierarchical RL."""

__all__ = ["DiffusionAugmentedBattleEnv", "DiffusionEnemyPredictor"]


def __getattr__(name):
    if name == "DiffusionEnemyPredictor":
        from exp_diffusion_hrl.inference.predictor import DiffusionEnemyPredictor

        return DiffusionEnemyPredictor
    if name == "DiffusionAugmentedBattleEnv":
        from exp_diffusion_hrl.env import DiffusionAugmentedBattleEnv

        return DiffusionAugmentedBattleEnv
    raise AttributeError(name)
