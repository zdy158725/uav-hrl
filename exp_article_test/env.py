import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from exp_article_test.config import HRLConfig, SelfPlayConfig
from exp_article_test.scene import BattleScene

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym


class BattleEnv(MultiAgentEnv):
    def __init__(self, num_blue=None, num_red=None, env_config=None):
        if env_config is None and isinstance(num_blue, dict):
            env_config = num_blue
            num_blue = None
            num_red = None

        env_config = env_config or {}
        self.num_blue = env_config.get("num_blue", num_blue if num_blue is not None else HRLConfig["start_train_num_blue"])
        self.num_red = env_config.get("num_red", num_red if num_red is not None else HRLConfig["start_train_num_red"])
        self.candidate_enemy_count = env_config.get("candidate_enemy_count", HRLConfig["candidate_enemy_count"])
        self.sub_steps = env_config.get("sub_steps", HRLConfig["sub_steps"])
        self.use_blue_script = env_config.get("use_blue_script", True)
        self.use_script_low_level = env_config.get("use_script_low_level", HRLConfig["use_script_low_level"])
        self.low_level_policy_dir = env_config.get("low_level_policy_dir", HRLConfig["low_level_policy_dir"])
        self.freeze_low_level = env_config.get("freeze_low_level", HRLConfig["freeze_low_level"])
        self.blue_policy_mode = env_config.get("blue_policy_mode", "script")
        self.blue_opponent_checkpoint = env_config.get("blue_opponent_checkpoint", None)
        self.opponent_pool_dir = env_config.get("opponent_pool_dir", SelfPlayConfig["opponent_pool_dir"])
        self.opponent_sample_mode = env_config.get(
            "opponent_sample_mode", SelfPlayConfig["opponent_sample_mode"]
        )
        self.fallback_to_script = env_config.get("fallback_to_script", SelfPlayConfig["fallback_to_script"])

        self.possible_agents = [f"red_{i}" for i in range(self.num_red)]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.possible_agents)

        self.scene = BattleScene(
            num_red=self.num_red,
            num_blue=self.num_blue,
            candidate_enemy_count=self.candidate_enemy_count,
            sub_steps=self.sub_steps,
            use_blue_script=self.use_blue_script,
            use_script_low_level=self.use_script_low_level,
            low_level_policy_dir=self.low_level_policy_dir,
            freeze_low_level=self.freeze_low_level,
            blue_policy_mode=self.blue_policy_mode,
            blue_opponent_checkpoint=self.blue_opponent_checkpoint,
            opponent_pool_dir=self.opponent_pool_dir,
            opponent_sample_mode=self.opponent_sample_mode,
            fallback_to_script=self.fallback_to_script,
        )
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.scene.high_level_obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.scene.action_dim)
        self.observation_spaces = {agent: self.observation_space for agent in self.agents}
        self.action_spaces = {agent: self.action_space for agent in self.agents}
        super().__init__()

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.scene.reset()
        obs = self.get_obs()
        self.agents = list(obs.keys())
        info = self.get_info()
        return obs, info

    def step(self, action):
        self.scene.step(action)
        obs = self.get_obs()
        reward = self.get_reward()
        terminated = self.get_terminated()
        truncated = self.get_truncated()
        info = self.get_info()
        self.agents = list(obs.keys())
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        return self.scene.observation()

    def get_reward(self):
        return self.scene.reward()

    def get_terminated(self):
        return self.scene.done()

    def get_truncated(self):
        truncated = {agent_id: False for agent_id in self.possible_agents}
        truncated["__all__"] = False
        return truncated

    def get_info(self):
        return self.scene.info()

    def call_back_winner(self):
        return self.scene.call_back_winner()

    def call_back_save_data(self):
        self.scene.call_back_save_data()

    def call_back_stats(self):
        return self.scene.get_episode_stats()
