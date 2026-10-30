import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from exp_article_test.config import HRLConfig
from exp_diffusion_hrl.config import DiffusionHRLConfig
from exp_diffusion_hrl.scene import DiffusionAugmentedBattleScene

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym


class DiffusionAugmentedBattleEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        config = env_config or {}
        self.num_red = int(config.get("num_red", DiffusionHRLConfig["num_red"]))
        self.num_blue = int(config.get("num_blue", DiffusionHRLConfig["num_blue"]))
        self.possible_agents = [f"red_{index}" for index in range(self.num_red)]
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        low_level_policy_dir = config.get("low_level_policy_dir")
        if not low_level_policy_dir:
            low_level_policy_dir = str(
                __import__("pathlib").Path(DiffusionHRLConfig["baseline_bundle"]) / "policies"
            )
        self.scene = DiffusionAugmentedBattleScene(
            num_red=self.num_red,
            num_blue=self.num_blue,
            candidate_enemy_count=config.get("candidate_enemy_count", 2),
            sub_steps=config.get("sub_steps", DiffusionHRLConfig["sub_steps"]),
            use_blue_script=True,
            use_script_low_level=config.get("use_script_low_level", False),
            low_level_policy_dir=low_level_policy_dir,
            freeze_low_level=True,
            blue_policy_mode=config.get("blue_policy_mode", "script"),
            blue_opponent_checkpoint=config.get("blue_opponent_checkpoint"),
            fallback_to_script=config.get("fallback_to_script", True),
            predictor_bundle=config.get("predictor_bundle", DiffusionHRLConfig["predictor_bundle"]),
            prediction_device=config.get("prediction_device", DiffusionHRLConfig["prediction_device"]),
            prediction_steps=config.get("prediction_steps", DiffusionHRLConfig["prediction_steps"]),
            num_inference_steps=config.get(
                "num_inference_steps", DiffusionHRLConfig["num_inference_steps"]
            ),
        )
        if self.scene.high_level_obs_dim != DiffusionHRLConfig["high_level_obs_dim"]:
            raise RuntimeError(f"Expected 58-dim observation, got {self.scene.high_level_obs_dim}")
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.scene.high_level_obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.scene.action_dim)
        self.observation_spaces = {agent: self.observation_space for agent in self.agents}
        self.action_spaces = {agent: self.action_space for agent in self.agents}
        super().__init__()

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.scene.reset()
        obs = self.scene.observation()
        self.agents = list(obs)
        return obs, self.scene.info()

    def step(self, actions):
        self.scene.step(actions)
        obs = self.scene.observation()
        rewards = self.scene.reward()
        terminated = self.scene.done()
        truncated = {agent: False for agent in self.possible_agents}
        truncated["__all__"] = False
        self.agents = list(obs)
        return obs, rewards, terminated, truncated, self.scene.info()

    def call_back_winner(self):
        return self.scene.call_back_winner()

    def call_back_save_data(self):
        self.scene.call_back_save_data()

    def call_back_stats(self):
        stats = self.scene.get_episode_stats()
        stats.update(
            {
                "prediction_ready_ratio": self.scene.prediction_metrics["ready_ratio"],
                "prediction_latency_ms": self.scene.prediction_metrics["latency_ms"],
            }
        )
        return stats
