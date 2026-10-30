import numpy as np

from exp_article_test.maneuver import select_candidate_targets
from exp_article_test.scene import BattleScene
from exp_diffusion_hrl.inference.predictor import DiffusionEnemyPredictor


class DiffusionAugmentedBattleScene(BattleScene):
    def __init__(
        self,
        *args,
        predictor_bundle,
        prediction_device="cuda",
        prediction_steps=(10, 20, 30),
        num_inference_steps=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.candidate_enemy_count != 2:
            raise ValueError("The first diffusion HRL version requires candidate_enemy_count=2")
        self.base_high_level_obs_dim = self.high_level_obs_dim
        self.prediction_steps = tuple(int(step) for step in prediction_steps)
        self.prediction_features_per_enemy = len(self.prediction_steps) * 3 + 1
        self.high_level_obs_dim += self.candidate_enemy_count * self.prediction_features_per_enemy
        self.predictor = DiffusionEnemyPredictor(
            artifact_dir=predictor_bundle,
            device=prediction_device,
            num_inference_steps=num_inference_steps,
        )
        self._prediction_cache_time = {}
        self._prediction_cache = {}
        self.prediction_metrics = {"ready_ratio": 0.0, "latency_ms": 0.0}

    def reset(self):
        super().reset()
        self._prediction_cache_time = {}
        self._prediction_cache = {}
        self.prediction_metrics = {"ready_ratio": 0.0, "latency_ms": 0.0}

    def _refresh_prediction_cache(self, enemy_team="Blue"):
        current_time = round(float(self.bf.t), 6)
        if current_time == self._prediction_cache_time.get(enemy_team):
            return
        self._prediction_cache[enemy_team] = self.predictor.predict(
            self.bf, target_team=enemy_team
        )
        self._prediction_cache_time[enemy_team] = current_time
        self.prediction_metrics = {
            "ready_ratio": float(self.predictor.last_ready_ratio),
            "latency_ms": float(self.predictor.last_latency_ms),
        }

    def get_high_level_obs(self, uav):
        base_obs = super().get_high_level_obs(uav)
        # Historical 38-dim blue policies remain valid when used as fixed opponents.
        if uav.team != "Red":
            return base_obs
        if not uav.alive:
            return np.zeros(self.high_level_obs_dim, dtype=np.float32)
        return self.get_augmented_high_level_obs(uav, base_obs=base_obs)

    def get_augmented_high_level_obs(self, uav, base_obs=None):
        if base_obs is None:
            base_obs = super().get_high_level_obs(uav)
        if not uav.alive:
            return np.zeros(self.high_level_obs_dim, dtype=np.float32)
        enemy_team = "Blue" if uav.team == "Red" else "Red"
        self._refresh_prediction_cache(enemy_team)
        prediction_cache = self._prediction_cache[enemy_team]
        candidates = select_candidate_targets(uav, self.bf, top_k=self.candidate_enemy_count)
        features = []
        for candidate_index in range(self.candidate_enemy_count):
            if candidate_index >= len(candidates):
                features.extend([0.0] * self.prediction_features_per_enemy)
                continue
            enemy = candidates[candidate_index]
            trajectory = prediction_cache.get(enemy.id)
            if trajectory is None:
                features.extend([0.0] * self.prediction_features_per_enemy)
                continue
            for step in self.prediction_steps:
                future_position = np.asarray(enemy.pos) + trajectory[step - 1]
                relative_position = future_position - np.asarray(uav.pos)
                features.extend(
                    [
                        relative_position[0] / 16000.0,
                        relative_position[1] / 16000.0,
                        relative_position[2] / 8000.0,
                    ]
                )
            ready = float(len(enemy.record_pos) >= self.predictor.history_steps)
            features.append(ready)
        return np.concatenate([base_obs, np.asarray(features, dtype=np.float32)]).astype(np.float32)

    def info(self):
        info = super().info()
        for agent_info in info.values():
            agent_info.update(
                {
                    "prediction_ready_ratio": self.prediction_metrics["ready_ratio"],
                    "prediction_latency_ms": self.prediction_metrics["latency_ms"],
                }
            )
        return info
