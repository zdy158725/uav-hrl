from ray.rllib.algorithms.callbacks import DefaultCallbacks


class _BaseBattleCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.red_win_counter = 0
        self.blue_win_counter = 0
        self.counter = 0

    def _record_common_metrics(self, episode, winner, red_alive, blue_alive):
        if winner == "red":
            self.red_win_counter += 1
        elif winner == "blue":
            self.blue_win_counter += 1
        self.counter += 1

        episode.custom_metrics["log_red_win_counter"] = self.red_win_counter
        episode.custom_metrics["log_blue_win_counter"] = self.blue_win_counter
        episode.custom_metrics["log_counter"] = self.counter
        episode.custom_metrics["log_red_win_mean"] = self.red_win_counter / self.counter
        episode.custom_metrics["log_blue_win_mean"] = self.blue_win_counter / self.counter
        episode.custom_metrics["log_red_alive"] = red_alive
        episode.custom_metrics["log_blue_alive"] = blue_alive
        episode.custom_metrics["log_episode_length"] = episode.length


class AlgoCallbacks(_BaseBattleCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        del worker, policies, kwargs
        env = base_env.get_sub_environments()[env_index]
        winner, red_alive, blue_alive = env.call_back_winner()
        step_stats = env.call_back_stats()

        if self.counter % 10000 == 57:
            env.call_back_save_data()

        self._record_common_metrics(episode, winner, red_alive, blue_alive)
        episode.custom_metrics["log_attack_ratio"] = step_stats["attack_ratio"]
        episode.custom_metrics["log_evade_ratio"] = step_stats["evade_ratio"]


class LowLevelCallbacks(_BaseBattleCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        del worker, policies, kwargs
        env = base_env.get_sub_environments()[env_index]
        winner, red_alive, blue_alive = env.call_back_winner()
        low_level_stats = env.call_back_low_level_stats()

        if self.counter % 10000 == 57:
            env.call_back_save_data()

        self._record_common_metrics(episode, winner, red_alive, blue_alive)
        episode.custom_metrics["log_low_level_kill_events"] = low_level_stats.get("kill_events", 0)
        episode.custom_metrics["log_low_level_death_events"] = low_level_stats.get("death_events", 0)
        episode.custom_metrics["log_low_level_out_events"] = low_level_stats.get("out_events", 0)
