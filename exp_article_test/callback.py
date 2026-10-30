from ray.rllib.algorithms.callbacks import DefaultCallbacks

from exp_article_test.maneuver import ACTION_COUNT


def _record_low_level_action_metrics(episode, stats):
    for action_id in range(ACTION_COUNT):
        metric_name = f"log_low_action_{action_id:02d}_ratio"
        episode.custom_metrics[metric_name] = stats.get(f"low_action_{action_id:02d}_ratio", 0.0)


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
        decisive_counter = self.red_win_counter + self.blue_win_counter
        episode.custom_metrics["log_red_win_no_draw_mean"] = (
            self.red_win_counter / decisive_counter if decisive_counter else 0.0
        )
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
        _record_low_level_action_metrics(episode, step_stats)


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
        _record_low_level_action_metrics(episode, low_level_stats)
